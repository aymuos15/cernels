import torch

from configs.base import Config
from kops.registry.gaussian_blur_2d import kernel as gblur_kernel


class GaussianBlur2d(Config):
    name = "gaussian_blur_2d"
    dtype = torch.float32
    op = "kornia.filters.gaussian_blur2d"
    kernel_size = (11, 11)
    sigma = (2.0, 2.0)
    custom = staticmethod(gblur_kernel)

    def inputs(self, device, dtype):
        b, c, h, w = 8, 3, 1024, 1024
        x = torch.randn(b, c, h, w, device=device, dtype=dtype)
        return x, self.kernel_size, self.sigma

    def baseline(self, x, kernel_size, sigma):
        import kornia

        return kornia.filters.gaussian_blur2d(x, kernel_size, sigma)
