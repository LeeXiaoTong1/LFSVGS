# import torch
#
# midas = torch.hub.load("intel-isl/MiDaS", "DPT_Hybrid")
# device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
# midas.to(device)
# midas.eval()
# for param in midas.parameters():
#     param.requires_grad = False
#
# midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
# transform = midas_transforms.dpt_transform
# downsampling = 1
#
# def estimate_depth(img, mode='test'):
#     h, w = img.shape[1:3]
#     norm_img = (img[None] - 0.5) / 0.5
#     norm_img = torch.nn.functional.interpolate(
#         norm_img,
#         size=(384, 512),
#         mode="bicubic",
#         align_corners=False)
#
#     if mode == 'test':
#         with torch.no_grad():
#             prediction = midas(norm_img)
#             prediction = torch.nn.functional.interpolate(
#                 prediction.unsqueeze(1),
#                 size=(h//downsampling, w//downsampling),
#                 mode="bicubic",
#                 align_corners=False,
#             ).squeeze()
#     else:
#         prediction = midas(norm_img)
#         prediction = torch.nn.functional.interpolate(
#             prediction.unsqueeze(1),
#             size=(h//downsampling, w//downsampling),
#             mode="bicubic",
#             align_corners=False,
#         ).squeeze()
#     return prediction
#

import cv2
import torch
import numpy as np
import os
from depth_anything_v2.dpt import DepthAnythingV2

def estimate_depth(img, model):
    # 确保图像是RGB格式
    if len(img.shape) == 2:  # 如果是灰度图，转换为RGB
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

    # 使用传入的模型进行深度推理
    depth = model.infer_image(img)  # HxW 深度图
    return depth
