"""Custom SiLU-and-mul: a hand-written CUDA kernel (silu_and_mul.cu), one pass.

JIT-compiled with torch's load_inline on first call (cached afterwards). Exposes
kernel(x) -> silu(x[..., :d]) * x[..., d:], matching the activation kernel.
"""

from importlib.resources import files
from typing import Any

from torch.utils.cpp_extension import load_inline

_DECL = "at::Tensor silu_and_mul(at::Tensor x);"
_SRC = files("kops").joinpath("silu_and_mul.cu").read_text()
_mod: Any = None


def kernel(x):
    global _mod
    if _mod is None:  # compile on first use, not at import
        _mod = load_inline(name="silu_and_mul_cuda", cpp_sources=_DECL, cuda_sources=_SRC, functions=["silu_and_mul"])
    return _mod.silu_and_mul(x)
