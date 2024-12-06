import torch
midas = torch.hub.load("intel-isl/MiDaS", "DPT_Hybrid")
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
midas.to(device)
midas.eval()
for param in midas.parameters():
    param.requires_grad = False

midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
transform = midas_transforms.dpt_transform
downsampling = 1

def estimate_depth(img, mode='test'):
    h, w = img.shape[1:3]
    norm_img = (img[None] - 0.5) / 0.5
    norm_img = torch.nn.functional.interpolate(
        norm_img,
        size=(384, 512),
        mode="bicubic",
        align_corners=False)

    if mode == 'test':
        with torch.no_grad():
            prediction = midas(norm_img)
            prediction = torch.nn.functional.interpolate(
                prediction.unsqueeze(1),
                size=(h//downsampling, w//downsampling),
                mode="bicubic",
                align_corners=False,
            ).squeeze()
    else:
        prediction = midas(norm_img)
        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size=(h//downsampling, w//downsampling),
            mode="bicubic",
            align_corners=False,
        ).squeeze()
    return prediction

# from depth_anything.dpt import DepthAnything
# from depth_anything.util.transform import Resize, NormalizeImage, PrepareForNet
# import cv2
# import torch
# from torchvision.transforms import Compose

# encoder = 'vitl' # can also be 'vitb' or 'vitl'
# depth_anything = DepthAnything.from_pretrained('LiheYoung/depth_anything_{:}14'.format(encoder)).eval()
# device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
# depth_anything.to(device)

# for param in depth_anything.parameters():
#     param.requires_grad = False

# # transform = Compose([
# #     Resize(
# #         width=518,
# #         height=518,
# #         resize_target=False,
# #         keep_aspect_ratio=True,
# #         ensure_multiple_of=14,
# #         resize_method='lower_bound',
# #         image_interpolation_method=cv2.INTER_CUBIC,
# #     ),
# #     NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
# #     PrepareForNet(),
# # ])


# def estimate_depth(img, mode='test'):
#     h, w = img.shape[1:3]
#     norm_img = (img[None] - 0.5) / 0.5
    
#     norm_img = torch.nn.functional.interpolate(
#         norm_img,
#         size=(518, 518),
#         mode="lower_bound",
#         align_corners=False)

#     if mode == 'test':
#         with torch.no_grad():
#             prediction = depth_anything(norm_img)
#             prediction = torch.nn.functional.interpolate(
#                 prediction.unsqueeze(1),
#                 size=(h, w),
#                 mode="lower_bound",
#                 align_corners=False,
#             ).squeeze()
#     else:
#         prediction = depth_anything(norm_img)
#         prediction = torch.nn.functional.interpolate(
#             prediction.unsqueeze(1),
#             size=(h, w),
#             mode="lower_bound",
#             align_corners=False,
#         ).squeeze()
#     return prediction
