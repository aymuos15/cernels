"""Custom RMSNorm: a hand-written CUDA kernel (rmsnorm.cu), one block per row.

Registered as a torch.library custom op (`kops::rmsnorm`) with a fake impl, so torch.compile
graphs *through* it with no graph break — the kernel composes with a compiled model. JIT-built
with load_inline on first call. kernel(x, weight, eps) -> normalized (matches F.rms_norm, bf16).
"""

from importlib.resources import files
from typing import Any

import torch
from torch import Tensor
from torch.utils.cpp_extension import load_inline

_DECL = "at::Tensor rmsnorm(at::Tensor x, at::Tensor weight, double eps);"
_SRC = files("kops.registry").joinpath("rmsnorm.cu").read_text()
_mod: Any = None


def _module():
    global _mod
    if _mod is None:  # compile on first use, not at import
        _mod = load_inline(name="rmsnorm_cuda", cpp_sources=_DECL, cuda_sources=_SRC, functions=["rmsnorm"])
    return _mod


@torch.library.custom_op("kops::rmsnorm", mutates_args=())
def kernel(x: Tensor, weight: Tensor, eps: float) -> Tensor:
    return _module().rmsnorm(x, weight, float(eps))


@kernel.register_fake
def _(x: Tensor, weight: Tensor, eps: float) -> Tensor:
    return torch.empty_like(x)
