"""
Copyright (c) 2025 Eastbean Zhang. All rights reserved.

This code is a modified version of the original work from:
https://github.com/graphdeco-inria/gaussian-splatting

"""

import torch
import torch.nn.functional as F
from torch.autograd import Variable
from math import exp
import pytorch3d
from utils.image_utils import erode
import numpy as np
from utils.mesh_utils import calculate_point_normals
def l1_loss(network_output, gt):
    return torch.abs((network_output - gt)).mean()

def l2_loss(network_output, gt):
    return ((network_output - gt) ** 2).mean()

def masked_l1_loss(network_output, gt, mask, bg_weight=0.1):
    # Foreground/background-weighted L1, normalized by weight.mean() so the
    # absolute loss scale matches vanilla l1_loss (= diff.mean()). Without
    # this normalization, image_loss shrinks while other lambda_* terms keep
    # their absolute scale, indirectly inflating their relative influence.
    weight = mask + (1.0 - mask) * bg_weight
    diff = (network_output - gt).abs()
    return (diff * weight).mean() / weight.mean().clamp(min=1e-6)

def masked_ssim(img1, img2, mask, window_size=11, bg_weight=0.1):
    # Same fg/bg weighting as masked_l1_loss applied to the SSIM map.
    # No mask dilation: the soft alpha mask naturally smooths across the
    # boundary, and the bg_weight floor preserves boundary monitoring.
    channel = img1.size(-3)
    window = create_window(window_size, channel)
    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    squeeze_back = (img1.dim() == 3)
    img1_b = img1.unsqueeze(0) if squeeze_back else img1
    img2_b = img2.unsqueeze(0) if squeeze_back else img2

    mu1 = F.conv2d(img1_b, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2_b, window, padding=window_size // 2, groups=channel)
    mu1_sq, mu2_sq, mu1_mu2 = mu1.pow(2), mu2.pow(2), mu1 * mu2
    sigma1_sq = F.conv2d(img1_b * img1_b, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2_b * img2_b, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1_b * img2_b, window, padding=window_size // 2, groups=channel) - mu1_mu2
    C1, C2 = 0.01 ** 2, 0.03 ** 2
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    mask_b = mask.unsqueeze(0) if mask.dim() == 3 else mask
    weight = mask_b + (1.0 - mask_b) * bg_weight
    return (ssim_map * weight).mean() / weight.mean().clamp(min=1e-6)

def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()

def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window

def ssim(img1, img2, window_size=11, size_average=True):
    channel = img1.size(-3)
    window = create_window(window_size, channel)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    return _ssim(img1, img2, window, window_size, channel, size_average)

def _ssim(img1, img2, window, window_size, channel, size_average=True):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)


def zero_one_loss(img):
    zero_epsilon = 1e-3
    val = torch.clamp(img, zero_epsilon, 1 - zero_epsilon)
    loss = torch.mean(torch.log(val) + torch.log(1 - val))
    return loss



def TVloss(img,mask=None):
    batch_size, c, h, w = img.size()
    if mask is not None:
        mask1=((mask[:,:,1:,:]+mask[:,:,:-1,:])/2)>0.99
        mask2=((mask[:,:,:,1:] + mask[:,:,:,:-1])/2)>0.99
        tv_h = torch.pow((img[:,:,1:,:] - img[:,:,:-1,:])*mask1, 2).sum()
        tv_w = torch.pow((img[:,:,:,1:] - img[:,:,:,:-1])*mask2, 2).sum()
    else:
        tv_h = torch.pow(img[:,:,1:,:] - img[:,:,:-1,:], 2).sum()
        tv_w = torch.pow(img[:,:,:,1:] - img[:,:,:,:-1], 2).sum()
    return (tv_h + tv_w) / (batch_size * c * h * w)


