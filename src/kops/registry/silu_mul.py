"""Custom fused SwiGLU activation (silu_mul.cu): out = silu(gate) * up, one pass (bf16).

Registered as a torch.library custom op (`kops::silu_mul`) with a fake impl, so torch.compile
graphs *through* it with no graph break. JIT-built with load_inline on first call.
kernel(gate, up) -> silu(gate)*up.
"""

from importlib.resources import files
from typing import Any

import torch
from torch import Tensor
from torch.utils.cpp_extension import load_inline

_DECL = "at::Tensor silu_mul(at::Tensor gate, at::Tensor up);"
_SRC = files("kops.registry").joinpath("silu_mul.cu").read_text()
_mod: Any = None


def _module():
    global _mod
    if _mod is None:
        _mod = load_inline(name="silu_mul_cuda", cpp_sources=_DECL, cuda_sources=_SRC, functions=["silu_mul"])
    return _mod


@torch.library.custom_op("kops::silu_mul", mutates_args=())
def kernel(gate: Tensor, up: Tensor) -> Tensor:
    return _module().silu_mul(gate, up)


@kernel.register_fake
def _(gate: Tensor, up: Tensor) -> Tensor:
    return torch.empty_like(gate)
