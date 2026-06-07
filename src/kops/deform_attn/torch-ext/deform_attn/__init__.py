from ._ops import ops  # type: ignore  # _ops is generated at build time


def ms_deform_attn_forward(
    value, spatial_shapes, level_start_index, sampling_locations, attention_weights, im2col_step
):
    return ops.ms_deform_attn_forward(
        value, spatial_shapes, level_start_index, sampling_locations, attention_weights, im2col_step
    )


__all__ = ["ms_deform_attn_forward"]
