#pragma once

#include <torch/torch.h>

at::Tensor multi_scale_deformable_attention(at::Tensor value, at::Tensor spatial_shapes, at::Tensor level_start_index,
                                            at::Tensor sampling_locations, at::Tensor attention_weights,
                                            int64_t im2col_step);
