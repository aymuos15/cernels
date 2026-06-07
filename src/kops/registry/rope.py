"""Custom RoPE: a hand-written CUDA kernel (rope.cu), one fused pass over q and k.

Registered as a torch.library custom op (`kops::rope`) with a fake impl so torch.compile graphs
through it. JIT-built with load_inline on first call. kernel(q, k, cos, sin) -> (q_out, k_out),
matching transformers' apply_rotary_pos_emb.
"""

from importlib.resources import files
from typing import Any

import torch
from torch import Tensor
from torch.utils.cpp_extension import load_inline

_DECL = "std::tuple<at::Tensor, at::Tensor> rope(at::Tensor q, at::Tensor k, at::Tensor cos, at::Tensor sin);"
_SRC = files("kops.registry").joinpath("rope.cu").read_text()
_mod: Any = None


def _module():
    global _mod
    if _mod is None:  # compile on first use, not at import
        _mod = load_inline(name="rope_cuda", cpp_sources=_DECL, cuda_sources=_SRC, functions=["rope"])
    return _mod


@torch.library.custom_op("kops::rope", mutates_args=())
def kernel(q: Tensor, k: Tensor, cos: Tensor, sin: Tensor) -> tuple[Tensor, Tensor]:
    return _module().rope(q, k, cos, sin)


@kernel.register_fake
def _(q: Tensor, k: Tensor, cos: Tensor, sin: Tensor) -> tuple[Tensor, Tensor]:
    return torch.empty_like(q), torch.empty_like(k)
