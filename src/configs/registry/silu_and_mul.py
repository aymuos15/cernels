import torch
import torch.nn.functional as F

from configs.base import Config
from kops.silu_and_mul import kernel as silu_and_mul_kernel


class SiluAndMul(Config):
    name = "silu_and_mul"
    repo = "kernels-community/activation"
    dtype = torch.float16
    op = "silu_and_mul"  # writes into a preallocated out of half the last dim
    out_arg = True
    custom = staticmethod(silu_and_mul_kernel)

    def inputs(self, device, dtype):
        return (torch.randn(4096, 8192, device=device, dtype=dtype),)

    def baseline(self, x):
        d = x.shape[-1] // 2
        return F.silu(x[..., :d]) * x[..., d:]

    def out_buffer(self, inputs):
        x = inputs[0]
        return torch.empty_like(x[..., : x.shape[-1] // 2])
