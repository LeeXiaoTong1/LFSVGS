#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
import torch
from torchmetrics import PearsonCorrCoef
from torchmetrics.functional.regression import pearson_corrcoef
from random import randint
from utils.loss_utils import l1_loss, l1_loss_mask, l2_loss, ssim, patch_norm_mse_loss
from depth_utils import estimate_depth
from gaussian_renderer import render, network_gui
import sys
from scene import Scene, GaussianModel
from utils.general_utils import safe_state, DecayScheduler
import uuid
from tqdm import tqdm
from utils.image_utils import psnr, downsample_image
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
from lpipsPyTorch import lpips
from scene.gaussian_model import build_scaling_rotation

import torch
import cv2
from depth_utils import estimate_depth
from depth_anything_v2.dpt import DepthAnythingV2

DEVICE = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'

# 加载模型
def load_depth_model(mode='vitl'):
    model_configs = {
        'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
        'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
        'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
        'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
    }

    model = DepthAnythingV2(**model_configs[mode])
    model.load_state_dict(torch.load(f'checkpoints/depth_anything_v2_{mode}.pth', map_location='cpu'))
    model = model.to(DEVICE).eval()
    return model
# 在训练开始时加载模型
depth_model = load_depth_model('vitl')



def save_depth_image(depth, filename):
    depth = depth.squeeze().detach().cpu().numpy()
    depth = (depth - depth.min()) / (depth.max() - depth.min())  # Normalize to [0, 1]
    depth = (depth * 255).astype(np.uint8)
    plt.imsave(filename, depth, cmap='gray')



def training(dataset, opt, pipe, args, depth_model):
    testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from, comp, store_npz = args.test_iterations, \
            args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args.debug_from, args.comp, args.store_npz
    
    first_iter = 0
    # resize_scale_sched = DecayScheduler(
    #                                     total_steps=int(opt.resize_period*(opt.iterations+1)),
    #                                     decay_name='cosine',
    #                                     start=opt.resize_scale,
    #                                     end=1.0,
    #                                     )
    
    tb_writer = prepare_output_and_logger(dataset)
    gaussians = GaussianModel(args)
    scene = Scene(args, gaussians, shuffle=False, depth_model=depth_model)
    gaussians.training_setup(opt)
    
    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)


    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing=True)
    iter_end = torch.cuda.Event(enable_timing=True)
    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")

    viewpoint_stack, pseudo_stack = None, None
    
    ema_loss_for_log = 0.0
    first_iter += 1
    for iteration in range(first_iter, opt.iterations + 1):
        if network_gui.conn == None:
            network_gui.try_connect()
        while network_gui.conn != None:
            try:
                net_image_bytes = None
                custom_cam, do_training, pipe.convert_SHs_python, pipe.compute_cov3D_python, keep_alive, scaling_modifer = network_gui.receive()
                if custom_cam != None:
                    net_image = render(custom_cam, gaussians, pipe, background, scaling_modifer)["render"]
                    net_image_bytes = memoryview((torch.clamp(net_image, min=0, max=1.0) * 255).byte().permute(1, 2, 0).contiguous().cpu().numpy())
                network_gui.send(net_image_bytes, dataset.source_path)
                if do_training and ((iteration < int(opt.iterations)) or not keep_alive):
                    break
            except Exception as e:
                network_gui.conn = None

        xyz_lr = gaussians.update_learning_rate(iteration)
        
        # Render
        if (iteration - 1) == debug_from:
            pipe.debug = True

        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 500 == 0:
            gaussians.oneupSHdegree()

        # Pick a random Camera
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack) - 1))


        # render_pkg = render(viewpoint_cam, gaussians, pipe, background)
        
        # Loss
        gt_image = viewpoint_cam.original_image.cuda()
        
#         if opt.transform == "downsample":
#             gt_image = downsample_image(gt_image, resize_scale_sched(iteration))
        
        
        if iteration <= opt.rvq_iter:
            render_pkg = render(viewpoint_cam, gaussians, pipe, background, image_shape=gt_image.shape, itr=iteration, rvq_iter=False)
        else:
            render_pkg = render(viewpoint_cam, gaussians, pipe, background, image_shape=gt_image.shape, itr=iteration, rvq_iter=False)
            
        image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]

        
        Ll1 =  l1_loss_mask(image, gt_image)
        # loss = ((1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(image, gt_image)))
        # loss = ((1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(image, gt_image))) + opt.lambda_mask*torch.mean((torch.sigmoid(gaussians._mask)))
        loss = ((1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(image, gt_image)))

        # loss = loss + args.opacity_reg * torch.abs(gaussians.get_opacity).mean()
        # loss = loss + args.scale_reg * torch.abs(gaussians.get_scaling).mean()

        rendered_depth = render_pkg["depth"][0]
        midas_depth = torch.tensor(viewpoint_cam.depth_image).cuda().squeeze()
        
#         lambda_normal = opt.lambda_normal if iteration > 2000 else 0.0
        
#         rendered_norm = render_pkg["normal"][0]
#         depth_normal = torch.tensor(viewpoint_cam.normal_map).cuda().squeeze()
#         normal = depth_normal[..., 0]
#         print(f"rendered_norm.shape: {rendered_norm.shape}")
#         print(f"normal.shape: {normal.shape}")

#         normal_error = (1 - (rendered_norm * normal).sum(dim=0))[None]
#         normal_error = torch.norm((rendered_norm - normal), dim=-1)
#         normal_loss = lambda_normal * (normal_error).mean()
#         loss = loss + normal_loss
        
        # 反转 rendered_depth 的深度值
        max_depth_value = rendered_depth.max()
        rendered_depth_inverted = max_depth_value - rendered_depth
        
        
#         print(rendered_depth.shape)
#         print('/////////////////////////////////////////////////////////////')
#         print(midas_depth.shape)
        
        

        # print(
        #     f"Iteration {iteration}: midas_depth shape: {midas_depth.shape}, rendered_depth shape: {rendered_depth.shape}")
        if iteration % 1000 == 0:
            save_depth_image(midas_depth, f'midas_depth_{iteration}.png')
            save_depth_image(rendered_depth_inverted, f'rendered_depth_{iteration}.png')

        # Ensure both depths have the same dimensions
        midas_depth_resized = torch.nn.functional.interpolate(
            midas_depth.unsqueeze(0).unsqueeze(0),
            size=rendered_depth.shape,
            mode='bicubic',
            align_corners=False
        ).squeeze()

        rendered_depth = rendered_depth.reshape(-1, 1)
        # rendered_depth_flat = rendered_depth_inverted.reshape(-1, 1)
        midas_depth_t = midas_depth_resized.reshape(-1, 1)
        
        

        depth_loss = min(
            (1 - pearson_corrcoef(-midas_depth_t, rendered_depth)),
            (1 - pearson_corrcoef(1 / (midas_depth_t + 200.), rendered_depth))
        )
        

        loss += args.depth_weight * depth_loss

        if iteration > args.end_sample_pseudo:
        # if iteration > 2000:
            args.depth_weight = 0.001
            
        import torch.nn.functional as F
        patch_range = (5, 17)
        
        depth_n = render_pkg["depth"][0].unsqueeze(0)
        anyth_n = midas_depth.unsqueeze(0)
        
        # 将 anyth_n 调整为与 depth_n 相同的尺度
        anyth_n = F.interpolate(anyth_n.unsqueeze(0), size=depth_n.shape[-2:], mode='bilinear', align_corners=False).squeeze(0)
        anyth_n = 255.0 - anyth_n
        
        loss_l2_dpt = patch_norm_mse_loss(depth_n[None,...], anyth_n[None,...], randint(patch_range[0], patch_range[1]), opt.error_tolerance)
        # loss += 0.08 * loss_l2_dpt
        loss += 0.01 * loss_l2_dpt


        loss.backward()

        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            if iteration % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}"})
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()
            
            if iteration == opt.iterations:
                storage = gaussians.final_prune(iteration, compress=comp)
                with open(os.path.join(args.model_path, "storage"), 'w') as c:
                    c.write(storage)
                gaussians.precompute()
            # assert hasattr(gaussians, '_feature'), "The _feature attribute has not been initialized."
            # Log and save
            training_report(tb_writer, iteration, Ll1, loss, l1_loss,
                            testing_iterations, scene, render, (pipe, background))

            if iteration > first_iter and (iteration in saving_iterations):
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration, compress=comp, store=store_npz)

            # if iteration > first_iter and (iteration in checkpoint_iterations):
            #     print("\n[ITER {}] Saving Checkpoint".format(iteration))
            #     torch.save((gaussians.capture(), iteration),
            #                scene.model_path + "/chkpnt" + str(iteration) + ".pth")
            

            #Densification
            if  iteration < opt.densify_until_iter:
                # Keep track of max radii in image-space for pruning
                gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)
            
                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    size_threshold = None
                    gaussians.densify_and_prune(opt.densify_grad_threshold, opt.prune_threshold, scene.cameras_extent, size_threshold, iteration)

            # Optimizer step
            if iteration < opt.iterations:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none = True)
                gaussians.optimizer_net.step()
                gaussians.optimizer_net.zero_grad(set_to_none = True)
                gaussians.scheduler_net.step()

            gaussians.update_learning_rate(iteration)
            if (iteration - args.start_sample_pseudo - 1) % opt.opacity_reset_interval == 0 and \
                    iteration > args.start_sample_pseudo:
                gaussians.reset_opacity()
            
            if iteration > first_iter and (iteration in checkpoint_iterations):
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")


def prepare_output_and_logger(args):
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str=os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])

    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer



def training_report(tb_writer, iteration, Ll1, loss, l1_loss, testing_iterations, scene : Scene, renderFunc, renderArgs):
    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/l1_loss', Ll1.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
        # tb_writer.add_scalar('iter_time', elapsed, iteration)

    # Report test and samples of training set
    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        validation_configs = ({'name': 'test', 'cameras' : scene.getTestCameras()},
                              {'name': 'train', 'cameras' : scene.getTrainCameras()})

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test, psnr_test, ssim_test, lpips_test = 0.0, 0.0, 0.0, 0.0
                for idx, viewpoint in enumerate(config['cameras']):
                    image = torch.clamp(renderFunc(viewpoint, scene.gaussians, *renderArgs)["render"], 0.0, 1.0)
                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    if tb_writer and (idx < 8):
                        tb_writer.add_images(config['name'] + "_view_{}/render".format(viewpoint.image_name), image[None], global_step=iteration)
                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name), gt_image[None], global_step=iteration)
                    l1_test += l1_loss(image, gt_image).mean().double()

                    _mask = None
                    _psnr = psnr(image, gt_image, _mask).mean().double()
                    _ssim = ssim(image, gt_image, _mask).mean().double()
                    _lpips = lpips(image, gt_image, _mask, net_type='vgg')
                    psnr_test += _psnr
                    ssim_test += _ssim
                    lpips_test += _lpips
                psnr_test /= len(config['cameras'])
                ssim_test /= len(config['cameras'])
                lpips_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])
                print("\n[ITER {}] Evaluating {}: L1 {} PSNR {} SSIM {} LPIPS {} ".format(
                    iteration, config['name'], l1_test, psnr_test, ssim_test, lpips_test))
                if tb_writer:
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)

        if tb_writer:
            tb_writer.add_histogram("scene/opacity_histogram", scene.gaussians.get_opacity, iteration)
            tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)
        torch.cuda.empty_cache()

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)

    parser.add_argument("--test_iterations", nargs="+", type=int, default=[10_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[10_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[10_000])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    parser.add_argument("--train_bg", action="store_true")
    parser.add_argument("--c2f", action="store_true", default=False)
    parser.add_argument("--c2f_every_step", type=float, default=1000,
                        help="Recompute low pass filter size for every c2f_every_step iterations")
    parser.add_argument("--c2f_max_lowpass", type=float, default=300, help="Maximum low pass filter size")
    parser.add_argument("--comp", action="store_true")
    parser.add_argument("--store_npz", action="store_true")
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)

    print(args.test_iterations)

    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Start GUI server, configure and run training
    # network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(lp.extract(args), op.extract(args), pp.extract(args), args, depth_model)

    # All done
    print("\nTraining complete.")
