"""Qwen3-Next Gated DeltaNet loader — kernel-builder kernel under
src/kops/qwen3_next_gated_deltanet/ (built with nix; see scripts/build_kernels.sh).

Exposes kernel(q, k, v, g, beta, chunk_size) -> core_attn_out (b, seq, num_heads, head_dim),
matching torch_chunk_gated_delta_rule (use_qk_l2norm_in_kernel=True, scale=1/sqrt(head_dim)).
The l2norm, cumulative gate decay, beta write-strength and the fp32 cross-chunk state update
all run inside the fused CUDA op — no Python prep.
"""

from pathlib import Path
from typing import Any

_mod: Any = None
_REPO = Path(__file__).resolve().parents[1] / "qwen3_next_gated_deltanet"


def _module():
    global _mod
    if _mod is None:
        from kernels import get_local_kernel

        _mod = get_local_kernel(_REPO)
    return _mod


def kernel(q, k, v, g, beta, chunk_size):
    return _module().deltanet(q, k, v, g, beta, int(chunk_size))
