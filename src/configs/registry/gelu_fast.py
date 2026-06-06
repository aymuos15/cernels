import torch
import torch.nn.functional as F

from configs.base import Config


class GeluFast(Config):
    name = "gelu_fast"
    repo = "kernels-community/activation"
    dtype = torch.float16
    op = "gelu_fast"
    out_arg = True

    def inputs(self, device, dtype):
        return (torch.randn(4096, 4096, device=device, dtype=dtype),)

    def baseline(self, x):
        return F.gelu(x)
