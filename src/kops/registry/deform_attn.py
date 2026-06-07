"""Custom fused CUDA kernel for multi-scale deformable attention.

One kernel pass: bilinear sample + attention weight multiply + accumulate. No per-level
intermediate tensors (unlike the grid_sample reference). The CUDA call is a torch.library
custom op (`kops::deform_attn`) with a fake impl so torch.compile graphs through it; the thin
kernel() wrapper drops the python spatial_shapes_list (a list, used only by the baseline).
-> Tensor shape (B, Q, n_heads*hidden).
"""

from importlib.resources import files
from typing import Any

import torch
from torch import Tensor
from torch.utils.cpp_extension import load_inline

_DECL = "at::Tensor ms_deform_attn_forward(at::Tensor value, at::Tensor spatial_shapes, at::Tensor level_start_index, at::Tensor sampling_locations, at::Tensor attention_weights, int im2col_step);"
_SRC = files("kops.registry").joinpath("deform_attn.cu").read_text()
_mod: Any = None


def _module():
    global _mod
    if _mod is None:
        _mod = load_inline(
            name="deform_attn_cuda", cpp_sources=_DECL, cuda_sources=_SRC, functions=["ms_deform_attn_forward"]
        )
    return _mod


@torch.library.custom_op("kops::deform_attn", mutates_args=())
def _forward(
    value: Tensor,
    spatial_shapes: Tensor,
    level_start_index: Tensor,
    sampling_locations: Tensor,
    attention_weights: Tensor,
    im2col_step: int,
) -> Tensor:
    return _module().ms_deform_attn_forward(
        value, spatial_shapes, level_start_index, sampling_locations, attention_weights, im2col_step
    )


@_forward.register_fake
def _(value, spatial_shapes, level_start_index, sampling_locations, attention_weights, im2col_step):
    b, _, n_heads, hidden = value.shape
    q = sampling_locations.shape[1]
    return value.new_empty((b, q, n_heads * hidden))


def kernel(
    value, spatial_shapes, spatial_shapes_list, level_start_index, sampling_locations, attention_weights, im2col_step=64
):
    # spatial_shapes_list (python list) is only for the transformers baseline; we use the tensor.
    return _forward(value, spatial_shapes, level_start_index, sampling_locations, attention_weights, im2col_step)
