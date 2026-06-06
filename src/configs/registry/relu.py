import torch

from configs.base import HubConfig


class Relu(HubConfig):
    name = "relu"
    repo = "kernels-community/relu"
    dtype = torch.float32
    op = "relu"
    out_arg = True

    def inputs(self, device, dtype):
        return (torch.randn(4096, 4096, device=device, dtype=dtype),)

    def baseline(self, x):
        return torch.relu(x)
