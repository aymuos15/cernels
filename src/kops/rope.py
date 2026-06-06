"""Custom RoPE: a hand-written CUDA kernel (rope.cu), one fused pass over q and k.

JIT-compiled with torch's load_inline on first call (cached afterwards). Exposes
kernel(q, k, cos, sin) -> (q_out, k_out), matching transformers' apply_rotary_pos_emb.
"""

from importlib.resources import files
from typing import Any

from torch.utils.cpp_extension import load_inline

_DECL = "std::tuple<at::Tensor, at::Tensor> rope(at::Tensor q, at::Tensor k, at::Tensor cos, at::Tensor sin);"
_SRC = files("kops").joinpath("rope.cu").read_text()
_mod: Any = None


def kernel(q, k, cos, sin):
    global _mod
    if _mod is None:  # compile on first use, not at import (keeps other benchmarks fast)
        _mod = load_inline(name="rope_cuda", cpp_sources=_DECL, cuda_sources=_SRC, functions=["rope"])
    return _mod.rope(q, k, cos, sin)
