"""silu_mul loader — kernel-builder kernel under src/kops/silu_mul/ (built with nix; see scripts/build_kernels.sh)."""

from pathlib import Path
from typing import Any

_mod: Any = None
_REPO = Path(__file__).resolve().parents[1] / "silu_mul"


def _module():
    global _mod
    if _mod is None:
        from kernels import get_local_kernel

        _mod = get_local_kernel(_REPO)
    return _mod


def kernel(gate, up):
    return _module().silu_mul(gate, up)
