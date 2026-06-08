"""gpt_oss_moe loader — kernel-builder kernel under src/kops/gpt_oss_moe/ (nix AOT).

The router (linear + top-k + softmax) runs HERE, in the timed path — only the raw
hidden state + the GptOssMLP module come from inputs(). The CUDA op does the token
routing, per-expert gate_up/down GEMMs (with biases) and the clamped-limited SwiGLU.
"""

from pathlib import Path
from typing import Any

import torch

_mod: Any = None
_REPO = Path(__file__).resolve().parents[1] / "gpt_oss_moe"


def _module():
    global _mod
    if _mod is None:
        from kernels import get_local_kernel

        _mod = get_local_kernel(_REPO)
    return _mod


def kernel(x, mlp):
    if mlp is None:
        raise RuntimeError("mlp module not available (meta device)")
    bsz, seqlen, hidden = x.shape
    xf = x.reshape(-1, hidden).contiguous()
    with torch.no_grad():
        _logits, scores, indices = mlp.router(xf)  # timed router (same code as baseline)
    e = mlp.experts
    out = _module().gpt_oss_moe(
        xf,
        e.gate_up_proj.data.contiguous(),
        e.gate_up_proj_bias.data.float().contiguous(),
        e.down_proj.data.contiguous(),
        e.down_proj_bias.data.float().contiguous(),
        indices.to(torch.int32).contiguous(),
        scores.float().contiguous(),
        mlp.router.top_k,
    )
    return out.reshape(bsz, seqlen, hidden)
