"""Custom NMS: a hand-written block-bitmask CUDA kernel (nms.cu).

Registered as a torch.library custom op (`kops::nms`) with a data-dependent fake (dynamic output
size) so torch.compile can trace it. JIT-built with load_inline on first call.
kernel(boxes, scores, iou) -> kept indices, matching torchvision.ops.nms.
"""

from importlib.resources import files
from typing import Any

import torch
from torch import Tensor
from torch.utils.cpp_extension import load_inline

_DECL = "at::Tensor nms(at::Tensor boxes, double thresh);"
_SRC = files("kops.registry").joinpath("nms.cu").read_text()
_mod: Any = None


def _module():
    global _mod
    if _mod is None:  # compile on first use, not at import
        _mod = load_inline(name="nms_cuda", cpp_sources=_DECL, cuda_sources=_SRC, functions=["nms"])
    return _mod


@torch.library.custom_op("kops::nms", mutates_args=())
def kernel(boxes: Tensor, scores: Tensor, iou: float) -> Tensor:
    order = scores.argsort(descending=True)
    keep = _module().nms(boxes[order].contiguous(), float(iou))  # indices into the sorted order
    return order[keep.to(order.device)]


@kernel.register_fake
def _(boxes: Tensor, scores: Tensor, iou: float) -> Tensor:
    ctx = torch.library.get_ctx()  # NMS output length is data-dependent
    return boxes.new_empty(ctx.new_dynamic_size(), dtype=torch.long)
