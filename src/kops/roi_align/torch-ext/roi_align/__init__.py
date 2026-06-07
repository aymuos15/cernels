from ._ops import ops  # type: ignore  # _ops is generated at build time


def roi_align(input, boxes, output_size, spatial_scale, sampling_ratio, aligned):
    return ops.roi_align(input, boxes, output_size, spatial_scale, sampling_ratio, aligned)


__all__ = ["roi_align"]
