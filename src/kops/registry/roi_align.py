"""roi_align loader — kernel-builder kernel under src/kops/roi_align/ (built with nix; see scripts/build_kernels.sh)."""

from pathlib import Path
from typing import Any

_mod: Any = None
_REPO = Path(__file__).resolve().parents[1] / "roi_align"


def _module():
    global _mod
    if _mod is None:
        from kernels import get_local_kernel

        _mod = get_local_kernel(_REPO)
    return _mod


def kernel(value, boxes, output_size=7, spatial_scale=1.0, sampling_ratio=2, aligned=True):
    return _module().roi_align(
        value.contiguous(), boxes.contiguous(), int(output_size), float(spatial_scale), int(sampling_ratio), aligned
    )
