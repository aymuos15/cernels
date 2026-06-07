"""Custom RoI Align kernel (roi_align.cu).

One thread per output element; bilinear-samples sampling_ratio^2 points per bin and
averages, matching torchvision.ops.roi_align semantics. fp32 accumulate, fp16 I/O.
JIT-compiled with load_inline on first call. Exposes
kernel(value, boxes, output_size, spatial_scale, sampling_ratio, aligned).
"""

from importlib.resources import files
from typing import Any

from torch.utils.cpp_extension import load_inline

_DECL = (
    "at::Tensor roi_align(at::Tensor input, at::Tensor boxes, int64_t output_size, "
    "double spatial_scale, int64_t sampling_ratio, bool aligned);"
)
_SRC = files("kops.registry").joinpath("roi_align.cu").read_text()
_mod: Any = None


def kernel(value, boxes, output_size=7, spatial_scale=1.0, sampling_ratio=2, aligned=True):
    global _mod
    if _mod is None:
        _mod = load_inline(name="roi_align_cuda", cpp_sources=_DECL, cuda_sources=_SRC, functions=["roi_align"])
    return _mod.roi_align(
        value.contiguous(), boxes.contiguous(), int(output_size), float(spatial_scale), int(sampling_ratio), aligned
    )
