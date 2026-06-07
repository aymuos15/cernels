"""Custom fused SwiGLU activation (silu_mul.cu): out = silu(gate) * up, one pass (bf16).

JIT-compiled with load_inline on first call. Exposes kernel(gate, up) -> silu(gate)*up.
"""

from importlib.resources import files
from typing import Any

from torch.utils.cpp_extension import load_inline

_DECL = "at::Tensor silu_mul(at::Tensor gate, at::Tensor up);"
_SRC = files("kops.registry").joinpath("silu_mul.cu").read_text()
_mod: Any = None


def kernel(gate, up):
    global _mod
    if _mod is None:
        _mod = load_inline(name="silu_mul_cuda", cpp_sources=_DECL, cuda_sources=_SRC, functions=["silu_mul"])
    return _mod.silu_mul(gate, up)
