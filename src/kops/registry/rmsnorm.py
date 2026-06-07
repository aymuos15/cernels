"""RMSNorm loader — the kernel is now a kernel-builder kernel under src/kops/rmsnorm/.

Built on the Spark with `nix run .#build-and-copy -L` (see scripts/build_kernels.sh), which
populates src/kops/rmsnorm/build/<variant>/. get_local_kernel picks the variant matching the
runtime. Exposes kernel(x, weight, eps) -> normalized, matching F.rms_norm (bf16).
"""

from pathlib import Path
from typing import Any

_mod: Any = None

_REPO = Path(__file__).resolve().parents[1] / "rmsnorm"  # src/kops/rmsnorm


def _module():
    global _mod
    if _mod is None:
        from kernels import get_local_kernel

        _mod = get_local_kernel(_REPO)
    return _mod


def kernel(x, weight, eps):
    return _module().rmsnorm(x, weight, float(eps))
