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
import matplotlib.pyplot as plt
import torch
import math
from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
from scene.gaussian_model import GaussianModel
from utils.sh_utils import eval_sh



def render(viewpoint_camera, pc: GaussianModel, pipe, bg_color : torch.Tensor, image_shape=None, scaling_modifier = 1.0,
           override_color = None, white_bg = False, itr=-1, rvq_iter=False):
    """
    Render the scene.

    Background tensor (bg_color) must be on GPU!
    """

    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass

    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)
                       
    if image_shape is None:
        image_shape = (3, viewpoint_camera.image_height, viewpoint_camera.image_width)

    if min(pc.bg_color.shape) != 0:
        bg_color = torch.tensor([0., 0., 0.]).cuda()

    confidence = pc.confidence if pipe.use_confidence else torch.ones_like(pc.confidence)
    raster_settings = GaussianRasterizationSettings(
        image_height=image_shape[1],
        image_width=image_shape[2],
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg = bg_color, #torch.tensor([1., 1., 1.]).cuda() if white_bg else torch.tensor([0., 0., 0.]).cuda(), #bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=pc.active_sh_degree,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=pipe.debug,
        confidence=confidence
    )
    
    
    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    means3D = pc.get_xyz
    means2D = screenspace_points
    cov3D_precomp = None
    # if pipe.compute_cov3D_python:
    #     cov3D_precomp = pc.get_covariance(scaling_modifier)
    # l_vqsca=0
    # l_vqrot=0
    if itr == -1:
        scales = pc._scaling
        rotations = pc._rotation
        opacity = pc._opacity
        
        dir_pp = (means3D - viewpoint_camera.camera_center.repeat(means3D.shape[0], 1))
        dir_pp = dir_pp/dir_pp.norm(dim=1, keepdim=True)
        shs = pc.mlp_head(torch.cat([pc._feature, pc.direction_encoding(dir_pp)], dim=-1)).unsqueeze(1)
        
    else:
        mask = ((torch.sigmoid(pc._mask) > 0.01).float()- torch.sigmoid(pc._mask)).detach() + torch.sigmoid(pc._mask)
        if rvq_iter:
            scales = pc.vq_scale(pc.get_scaling.unsqueeze(0))[0]
            rotations = pc.vq_rot(pc.get_rotation.unsqueeze(0))[0]
            scales = scales.squeeze()*mask
            rotations = rotations.squeeze()
            opacity = pc.get_opacity*mask
        else:
            scales = pc.get_scaling*mask
            rotations = pc.get_rotation
            opacity = pc.get_opacity*mask
            
        xyz = pc.contract_to_unisphere(means3D.clone().detach(), torch.tensor([-1.0, -1.0, -1.0, 1.0, 1.0, 1.0], device='cuda'))
        dir_pp = (means3D - viewpoint_camera.camera_center.repeat(means3D.shape[0], 1))
        dir_pp = dir_pp/dir_pp.norm(dim=1, keepdim=True)
        shs = pc.mlp_head(torch.cat([pc.recolor(xyz), pc.direction_encoding(dir_pp)], dim=-1)).unsqueeze(1)


    # Rasterize visible Gaussians to image, obtain their radii (on screen).
    rendered_image, radii, depth, alpha = rasterizer(
        means3D = means3D.float(),
        means2D = means2D,
        shs = shs.float(),
        colors_precomp = None,
        opacities = opacity,
        scales = scales,
        rotations = rotations,
        cov3D_precomp = cov3D_precomp)

    # rendered_image_list, depth_list, alpha_list = [], [], []
    # for i in range(5):
    #     rendered_image, radii, depth, alpha = rasterizer(
    #         means3D=means3D,
    #         means2D=means2D,
    #         shs=shs,
    #         colors_precomp=colors_precomp,
    #         opacities=opacity,
    #         scales=scales,
    #         rotations=rotations,
    #         cov3D_precomp=cov3D_precomp)
    #     rendered_image_list.append(rendered_image)
    #     depth_list.append(depth)
    #     alpha_list.append(alpha)
    # def mean1(t):
    #     return torch.mean(torch.stack(t), 0)
    # rendered_image, depth, alpha = mean1(rendered_image_list), mean1(depth_list), mean1(alpha_list)

    if min(pc.bg_color.shape) != 0:
        rendered_image = rendered_image + (1 - alpha) * torch.sigmoid(pc.bg_color)  # torch.ones((3, 1, 1)).cuda()


    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.
    return {"render": rendered_image,
            "viewspace_points": screenspace_points,
            "visibility_filter" : radii > 0,
            "radii": radii,
            "depth": depth}
