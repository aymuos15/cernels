"""gaussian_blur_2d loader."""

import torch


from kops.registry._local import load


def _k1d(ks, sigma, device):
    x = torch.arange(ks, device=device, dtype=torch.float32) - ks // 2
    g = torch.exp(-(x**2) / (2.0 * sigma**2))
    return g / g.sum()


def kernel(x, kernel_size, sigma):
    kh, kw = kernel_size
    sy, sx = sigma
    return load("gaussian_blur_2d").gaussian_blur_2d(x.contiguous(), _k1d(kh, sy, x.device), _k1d(kw, sx, x.device))
