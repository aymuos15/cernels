from ._ops import ops  # type: ignore  # _ops is generated at build time


def multi_scale_deformable_attention(
    value, spatial_shapes, level_start_index, sampling_locations, attention_weights, im2col_step
):
    return ops.multi_scale_deformable_attention(
        value, spatial_shapes, level_start_index, sampling_locations, attention_weights, im2col_step
    )


__all__ = ["multi_scale_deformable_attention"]
