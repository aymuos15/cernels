"""rope3d loader — kernel-builder kernel under src/kops/rope3d/ (built with nix; see scripts/build_kernels.sh)."""

from pathlib import Path
from typing import Any

_mod: Any = None
_REPO = Path(__file__).resolve().parents[1] / "rope3d"


def _module():
    global _mod
    if _mod is None:
        from kernels import get_local_kernel

        _mod = get_local_kernel(_REPO)
    return _mod


def kernel(q, k, emb):
    return _module().rope3d(q, k, emb)
