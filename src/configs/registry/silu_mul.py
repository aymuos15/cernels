"""SwiGLU activation: reference is the canonical torch composition F.silu(gate) * up (the op IS
the reference). Custom is our fused bf16 CUDA kernel (kops/registry/silu_mul.cu). Same kernel used
in the modelkernels integration.

Timed path: the fused activation. Precomputed in inputs(): the raw gate / up tensors.
"""

import torch
import torch.nn.functional as F

from configs.base import Config
from kops.registry.silu_mul import kernel as silu_mul_kernel


class SiluMul(Config):
    name = "silu_mul"
    dtype = torch.bfloat16
    op = "F.silu(gate) * up"
    custom = staticmethod(silu_mul_kernel)

    _M = 4096  # tokens
    _F = 5632  # ffn hidden

    def inputs(self, device, dtype):
        gate = torch.randn(self._M, self._F, device=device, dtype=dtype)
        up = torch.randn(self._M, self._F, device=device, dtype=dtype)
        return gate, up

    def baseline(self, gate, up):
        return F.silu(gate) * up

    def verify(self, out, ref) -> bool:
        return bool(torch.allclose(out.float(), ref.float(), atol=2e-2, rtol=2e-2))
