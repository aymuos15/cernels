import torch

from configs.base import Config
from kops.registry.gaussian_blur import kernel as gblur_kernel


class GaussianBlur(Config):
    name = "gaussian_blur"
    dtype = torch.float32
    op = "kornia.filters.gaussian_blur2d"
    kernel_size = (11, 11)
    sigma = (2.0, 2.0)
    custom = staticmethod(gblur_kernel)

    def inputs(self, device, dtype):
        b, c, h, w = 8, 3, 1024, 1024
        x = torch.randn(b, c, h, w, device=device, dtype=dtype)
        return x, self.kernel_size, self.sigma

    def baseline(self, x, kernel_size, sigma):  # the op is the reference (eager workload)
        import kornia

        return kornia.filters.gaussian_blur2d(x, kernel_size, sigma)
