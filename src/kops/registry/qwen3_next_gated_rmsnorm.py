"""qwen3_next_gated_rmsnorm loader — kernel-builder kernel under
src/kops/qwen3_next_gated_rmsnorm/ (built with nix; see scripts/build_kernels.sh).

Exposes kernel(hidden_states, gate, weight, eps) -> fused gated RMSNorm output (bf16),
matching transformers Qwen3NextRMSNormGated.forward (norm-before-gate, plain weight).
"""

from pathlib import Path
from typing import Any

_mod: Any = None
_REPO = Path(__file__).resolve().parents[1] / "qwen3_next_gated_rmsnorm"


def _module():
    global _mod
    if _mod is None:
        from kernels import get_local_kernel

        _mod = get_local_kernel(_REPO)
    return _mod


def kernel(hidden_states, gate, weight, eps):
    return _module().gated_rmsnorm(hidden_states, gate, weight, float(eps))
