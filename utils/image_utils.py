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

import torch
import numpy as np
import torchvision
import torch.nn as nn

def downsample_image(original_image, scale=1.0):

    assert scale<=1.0, "Scale must be <= 1.0"
    if scale == 1.0:
        return original_image
    else:
        return nn.functional.interpolate(original_image.unsqueeze(0), scale_factor=scale, mode='bilinear', align_corners=False).squeeze(0)

def mse(img1, img2):
    return (((img1 - img2)) ** 2).view(img1.shape[0], -1).mean(1, keepdim=True)

def psnr(img1, img2, mask=None):
    if mask is None:
        mse = (((img1 - img2)) ** 2).view(img1.shape[0], -1).mean(1, keepdim=True)
    else:
        mask_bin = (mask == 1.)
        mse = (((img1 - img2)[mask_bin]) ** 2).mean()
    return 20 * torch.log10(1.0 / torch.sqrt(mse))
