"""Custom 3D axial RoPE kernel: fused CUDA pass over q and k (rope3d.cu).

Uses timm's cat-style interleaved-pair layout (rot() function):
  out[2i]   = x[2i] * cos[i] - x[2i+1] * sin[i]
  out[2i+1] = x[2i+1] * cos[i] + x[2i] * sin[i]

JIT-compiled with load_inline on first call. Exposes kernel(q, k, emb) -> (q_out, k_out).
"""

from importlib.resources import files
from typing import Any

from torch.utils.cpp_extension import load_inline

_DECL = "std::tuple<at::Tensor, at::Tensor> rope3d(at::Tensor q, at::Tensor k, at::Tensor emb);"
_SRC = files("kops.registry").joinpath("rope3d.cu").read_text()
_mod: Any = None


def kernel(q, k, emb):
    global _mod
    if _mod is None:
        _mod = load_inline(name="rope3d_cuda", cpp_sources=_DECL, cuda_sources=_SRC, functions=["rope3d"])
    return _mod.rope3d(q, k, emb)
