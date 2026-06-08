"""RmsNorm: reference is torch.nn.functional.rms_norm (the op IS the reference).

Timed path: the normalization over the last dim. Precomputed in inputs(): the raw tensor + weight.
"""

import torch

from configs.base import Config
from kops.registry.rms_norm import kernel as rmsnorm_kernel


class RmsNorm(Config):
    name = "rms_norm"
    dtype = torch.bfloat16
    op = "torch.nn.functional.rms_norm"
    custom = staticmethod(rmsnorm_kernel)

    _M = 4096  # tokens
    _H = 2048  # hidden
    _eps = 1e-6

    def inputs(self, device, dtype):
        x = torch.randn(self._M, self._H, device=device, dtype=dtype)
        weight = torch.randn(self._H, device=device, dtype=dtype)
        return x, weight, self._eps

    def baseline(self, x, weight, eps):
        return torch.nn.functional.rms_norm(x, weight.shape, weight, eps)

    def verify(self, out, ref) -> bool:
        return bool(torch.allclose(out.float(), ref.float(), atol=2e-2, rtol=2e-2))
