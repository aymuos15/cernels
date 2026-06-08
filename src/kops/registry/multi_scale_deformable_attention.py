"""multi_scale_deformable_attention loader."""

from kops.registry._local import load


def kernel(
    value, spatial_shapes, spatial_shapes_list, level_start_index, sampling_locations, attention_weights, im2col_step=64
):
    return load("multi_scale_deformable_attention").multi_scale_deformable_attention(
        value, spatial_shapes, level_start_index, sampling_locations, attention_weights, im2col_step
    )
