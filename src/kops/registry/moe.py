"""Custom fused grouped-GEMM MoE kernel (moe.cu).

JIT-compiled with torch's load_inline on first call (cached afterwards). Exposes
kernel(x, moe) taking the same inputs as the megablocks_moe config baseline, running:
  - Router: x @ router_w^T + softmax + top-k=2 (in PyTorch with bf16, matching megablocks)
  - Token gather/permute per expert (CUDA)
  - Per-expert GEMM (w1 -> activation -> w2) with fp32 accumulation (CUDA)
  - Weighted scatter-combine back to [T, H] (CUDA)

Timed path includes: router linear + softmax + topk, weight extraction, per-expert CUDA GEMMs,
gather, activation, scatter. All of these run on every benchmark iteration.

Routing is done in PyTorch (same dtype as megablocks, bf16 linear) to match routing decisions
exactly. The large GEMM work (w1 @ x, w2 @ h1 per expert, 8 experts x 2 passes) is in CUDA.
"""

from importlib.resources import files
from typing import Any

import torch
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
_ACT_IDS = {
    "gelu": 0,
    "gelu_new": 0,
    "gelu_fast": 0,
    "silu": 1,
    "swish": 1,
    "relu": 2,
}


def kernel(x, moe):
    """Custom MoE forward.  Takes the same (x, moe) inputs as the megablocks baseline."""
    global _mod
    if _mod is None:
        _mod = load_inline(
            name="moe_cuda",
            cpp_sources=_DECL,
            cuda_sources=_SRC,
            functions=["moe_grouped_gemm"],
            extra_ldflags=["-lcublas"],
        )

    if moe is None:
        raise RuntimeError("moe module not available (meta device)")

    # --- Routing: call megablocks router directly for exact numerical match ---
    # This ensures our routing decisions are bit-for-bit identical to the reference.
    # The router call (linear + softmax + topk over E=8 experts) is fast relative to the
    # large H×F GEMMs and is part of the timed path.
    topk = moe.experts.args.moe_top_k
    with torch.no_grad():
        x3d = x.unsqueeze(1)  # router expects [sl, bs, hs]
        _scores, expert_weights_bf16, expert_indices = moe.router(x3d)
    # expert_weights_bf16: [T, topk] bf16, expert_indices: [T, topk] int64
    indices = expert_indices.to(torch.int32)  # [T, topk] int32
    wts = expert_weights_bf16.float()  # [T, topk] fp32

    # --- Expert weights ---
    w1 = moe.experts.mlp.w1.data.contiguous()  # [E, H, F] bf16
    w2 = moe.experts.mlp.w2.data.contiguous()  # [E, F, H] bf16

    # --- Activation ID ---
    act_fn = getattr(moe.experts.args, "activation_fn", "gelu")
    if callable(act_fn):
        act_fn_name = getattr(act_fn, "__name__", "gelu")
    else:
        act_fn_name = str(act_fn)
    act_id = _ACT_IDS.get(act_fn_name.lower(), 0)

    x_cont = x.contiguous()
    return _mod.moe_grouped_gemm(x_cont, w1, w2, indices.contiguous(), wts.contiguous(), topk, act_id)
