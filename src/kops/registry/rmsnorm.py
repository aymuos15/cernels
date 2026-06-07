"""Custom RMSNorm: a hand-written CUDA kernel (rmsnorm.cu), one block per row.

JIT-compiled with torch's load_inline on first call (cached afterwards). Exposes
kernel(x, weight, eps) -> normalized, matching Lfm2RMSNorm / F.rms_norm semantics (bf16).
"""

from importlib.resources import files
from typing import Any

from torch.utils.cpp_extension import load_inline

_DECL = "at::Tensor rmsnorm(at::Tensor x, at::Tensor weight, double eps);"
_SRC = files("kops.registry").joinpath("rmsnorm.cu").read_text()
_mod: Any = None


def kernel(x, weight, eps):
    global _mod
    if _mod is None:  # compile on first use, not at import
        _mod = load_inline(name="rmsnorm_cuda", cpp_sources=_DECL, cuda_sources=_SRC, functions=["rmsnorm"])
    return _mod.rmsnorm(x, weight, float(eps))
