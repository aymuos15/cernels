"""Custom fused CUDA kernel for multi-scale deformable attention.

One kernel pass: bilinear sample + attention weight multiply + accumulate.
No per-level intermediate tensors (unlike the grid_sample reference).

Exposes kernel(value, spatial_shapes, level_start_index, sampling_locations,
               attention_weights, im2col_step) -> Tensor  shape (B, Q, n_heads*hidden).
"""

from importlib.resources import files
from typing import Any

from torch.utils.cpp_extension import load_inline

_DECL = "at::Tensor ms_deform_attn_forward(at::Tensor value, at::Tensor spatial_shapes, at::Tensor level_start_index, at::Tensor sampling_locations, at::Tensor attention_weights, int im2col_step);"
_SRC = files("kops.registry").joinpath("deform_attn.cu").read_text()
_mod: Any = None


def kernel(
    value, spatial_shapes, spatial_shapes_list, level_start_index, sampling_locations, attention_weights, im2col_step=64
):
    # spatial_shapes_list is the Python list version (for the transformers baseline); we use
    # the tensor spatial_shapes.  im2col_step is an upstream batching hint; ignored here.
    global _mod
    if _mod is None:
        _mod = load_inline(
            name="deform_attn_cuda",
            cpp_sources=_DECL,
            cuda_sources=_SRC,
            functions=["ms_deform_attn_forward"],
        )
    return _mod.ms_deform_attn_forward(
        value, spatial_shapes, level_start_index, sampling_locations, attention_weights, im2col_step
    )
