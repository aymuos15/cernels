"""Custom RoI Align kernel (roi_align.cu).

One thread per output element; bilinear-samples sampling_ratio^2 points per bin and
averages, matching torchvision.ops.roi_align semantics. fp32 accumulate, fp16 I/O.
Registered as a torch.library custom op (`kops::roi_align`) with a fake impl so torch.compile
graphs through it. JIT-built with load_inline on first call.
kernel(value, boxes, output_size, spatial_scale, sampling_ratio, aligned).
"""

from importlib.resources import files
from typing import Any

import torch
from torch import Tensor
from torch.utils.cpp_extension import load_inline

_DECL = (
    "at::Tensor roi_align(at::Tensor input, at::Tensor boxes, int64_t output_size, "
    "double spatial_scale, int64_t sampling_ratio, bool aligned);"
)
_SRC = files("kops.registry").joinpath("roi_align.cu").read_text()
_mod: Any = None


def _module():
    global _mod
    if _mod is None:
        _mod = load_inline(name="roi_align_cuda", cpp_sources=_DECL, cuda_sources=_SRC, functions=["roi_align"])
    return _mod


@torch.library.custom_op("kops::roi_align", mutates_args=())
def kernel(
    value: Tensor,
    boxes: Tensor,
    output_size: int = 7,
    spatial_scale: float = 1.0,
    sampling_ratio: int = 2,
    aligned: bool = True,
) -> Tensor:
    return _module().roi_align(
        value.contiguous(), boxes.contiguous(), int(output_size), float(spatial_scale), int(sampling_ratio), aligned
    )


@kernel.register_fake
def _(
    value: Tensor,
    boxes: Tensor,
    output_size: int = 7,
    spatial_scale: float = 1.0,
    sampling_ratio: int = 2,
    aligned: bool = True,
) -> Tensor:
    return value.new_empty((boxes.shape[0], value.shape[1], output_size, output_size))
