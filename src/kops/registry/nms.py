"""Custom NMS: a hand-written block-bitmask CUDA kernel (nms.cu).

JIT-compiled with torch's load_inline on first call (cached afterwards). Exposes
kernel(boxes, scores, iou) -> kept indices, matching torchvision.ops.nms.
"""

from importlib.resources import files
from typing import Any

from torch.utils.cpp_extension import load_inline

_DECL = "at::Tensor nms(at::Tensor boxes, double thresh);"
_SRC = files("kops.registry").joinpath("nms.cu").read_text()
_mod: Any = None


def kernel(boxes, scores, iou):
    global _mod
    if _mod is None:  # compile on first use, not at import
        _mod = load_inline(name="nms_cuda", cpp_sources=_DECL, cuda_sources=_SRC, functions=["nms"])
    order = scores.argsort(descending=True)
    keep = _mod.nms(boxes[order].contiguous(), float(iou))  # CPU indices into the sorted order
    return order[keep.to(order.device)]
