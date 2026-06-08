"""roi_align loader."""

from kops.registry._local import load


def kernel(value, boxes, output_size=7, spatial_scale=1.0, sampling_ratio=2, aligned=True):
    return load("roi_align").roi_align(
        value.contiguous(), boxes.contiguous(), int(output_size), float(spatial_scale), int(sampling_ratio), aligned
    )
