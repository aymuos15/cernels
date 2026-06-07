"""Custom fused grouped-GEMM MoE kernel (moe.cu).

The grouped GEMM is a torch.library custom op (`kops::moe_grouped_gemm`) with a fake impl so
torch.compile graphs through it; the thin kernel(x, moe) wrapper does routing (calls the
megablocks router for an exact match) and weight extraction, then calls the custom op:
  - Router: x @ router_w^T + softmax + top-k=2 (PyTorch bf16, matching megablocks) [wrapper]
  - Per-expert GEMM (w1 -> activation -> w2) fp32-accumulate + gather/scatter-combine [custom op]
JIT-built with load_inline on first call.
"""

from importlib.resources import files
from typing import Any

import torch
from torch import Tensor
from torch.utils.cpp_extension import load_inline

_DECL = (
    "at::Tensor moe_grouped_gemm("
    "at::Tensor x, at::Tensor w1, at::Tensor w2, "
    "at::Tensor indices, at::Tensor weights, "
    "int topk, int act_id);"
)
_SRC = files("kops.registry").joinpath("moe.cu").read_text()
_mod: Any = None

# Map activation function names to integer IDs used in the CUDA kernel
_ACT_IDS = {"gelu": 0, "gelu_new": 0, "gelu_fast": 0, "silu": 1, "swish": 1, "relu": 2}


def _module():
    global _mod
    if _mod is None:
        _mod = load_inline(
            name="moe_cuda",
            cpp_sources=_DECL,
            cuda_sources=_SRC,
            functions=["moe_grouped_gemm"],
            extra_ldflags=["-lcublas"],
        )
    return _mod


@torch.library.custom_op("kops::moe_grouped_gemm", mutates_args=())
def _grouped_gemm(
    x: Tensor, w1: Tensor, w2: Tensor, indices: Tensor, weights: Tensor, topk: int, act_id: int
) -> Tensor:
    return _module().moe_grouped_gemm(x, w1, w2, indices, weights, topk, act_id)


@_grouped_gemm.register_fake
def _(x: Tensor, w1: Tensor, w2: Tensor, indices: Tensor, weights: Tensor, topk: int, act_id: int) -> Tensor:
    return torch.empty_like(x)


def kernel(x, moe):
    """Custom MoE forward. Takes the same (x, moe) inputs as the megablocks baseline."""
    if moe is None:
        raise RuntimeError("moe module not available (meta device)")

    # Routing via the megablocks router for a bit-for-bit match (part of the timed path).
    topk = moe.experts.args.moe_top_k
    with torch.no_grad():
        _scores, expert_weights_bf16, expert_indices = moe.router(x.unsqueeze(1))  # router wants [sl, bs, hs]
    indices = expert_indices.to(torch.int32).contiguous()  # [T, topk]
    wts = expert_weights_bf16.float().contiguous()  # [T, topk]

    w1 = moe.experts.mlp.w1.data.contiguous()  # [E, H, F] bf16
    w2 = moe.experts.mlp.w2.data.contiguous()  # [E, F, H] bf16

    act_fn = getattr(moe.experts.args, "activation_fn", "gelu")
    act_fn_name = getattr(act_fn, "__name__", "gelu") if callable(act_fn) else str(act_fn)
    act_id = _ACT_IDS.get(act_fn_name.lower(), 0)

    return _grouped_gemm(x.contiguous(), w1, w2, indices, wts, topk, act_id)
