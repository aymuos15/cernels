"""Custom 3D axial RoPE kernel: fused CUDA pass over q and k (rope3d.cu).

Uses timm's cat-style interleaved-pair layout (rot() function):
  out[2i]   = x[2i] * cos[i] - x[2i+1] * sin[i]
  out[2i+1] = x[2i+1] * cos[i] + x[2i] * sin[i]

Registered as a torch.library custom op (`kops::rope3d`) with a fake impl so torch.compile graphs
through it. JIT-built with load_inline on first call. kernel(q, k, emb) -> (q_out, k_out).
"""

from importlib.resources import files
from typing import Any

import torch
from torch import Tensor
from torch.utils.cpp_extension import load_inline

_DECL = "std::tuple<at::Tensor, at::Tensor> rope3d(at::Tensor q, at::Tensor k, at::Tensor emb);"
_SRC = files("kops.registry").joinpath("rope3d.cu").read_text()
_mod: Any = None


def _module():
    global _mod
    if _mod is None:
        _mod = load_inline(name="rope3d_cuda", cpp_sources=_DECL, cuda_sources=_SRC, functions=["rope3d"])
    return _mod


@torch.library.custom_op("kops::rope3d", mutates_args=())
def kernel(q: Tensor, k: Tensor, emb: Tensor) -> tuple[Tensor, Tensor]:
    return _module().rope3d(q, k, emb)


@kernel.register_fake
def _(q: Tensor, k: Tensor, emb: Tensor) -> tuple[Tensor, Tensor]:
    return torch.empty_like(q), torch.empty_like(k)
