"""nms loader — kernel-builder kernel under src/kops/nms/ (built with nix; see scripts/build_kernels.sh)."""

from pathlib import Path
from typing import Any

_mod: Any = None
_REPO = Path(__file__).resolve().parents[1] / "nms"


def _module():
    global _mod
    if _mod is None:
        from kernels import get_local_kernel

        _mod = get_local_kernel(_REPO)
    return _mod


def kernel(boxes, scores, iou):
    order = scores.argsort(descending=True)
    keep = _module().nms(boxes[order].contiguous(), float(iou))
    return order[keep.to(order.device)]
