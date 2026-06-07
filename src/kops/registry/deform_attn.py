"""deform_attn loader — kernel-builder kernel under src/kops/deform_attn/ (built with nix; see scripts/build_kernels.sh)."""

from pathlib import Path
from typing import Any

_mod: Any = None
_REPO = Path(__file__).resolve().parents[1] / "deform_attn"


def _module():
    global _mod
    if _mod is None:
        from kernels import get_local_kernel

        _mod = get_local_kernel(_REPO)
    return _mod


def kernel(
    value, spatial_shapes, spatial_shapes_list, level_start_index, sampling_locations, attention_weights, im2col_step=64
):
    return _module().ms_deform_attn_forward(
        value, spatial_shapes, level_start_index, sampling_locations, attention_weights, im2col_step
    )
