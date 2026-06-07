#pragma once

#include <torch/torch.h>

at::Tensor roi_align(at::Tensor input, at::Tensor boxes, int64_t output_size, double spatial_scale,
                     int64_t sampling_ratio, bool aligned);
