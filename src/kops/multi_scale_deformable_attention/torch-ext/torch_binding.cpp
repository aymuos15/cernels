#include <torch/library.h>

#include "registration.h"
#include "torch_binding.h"

TORCH_LIBRARY_EXPAND(TORCH_EXTENSION_NAME, ops) {
    ops.def("multi_scale_deformable_attention(Tensor value, Tensor spatial_shapes, Tensor level_start_index, Tensor "
            "sampling_locations, Tensor attention_weights, int im2col_step) -> Tensor");
    ops.impl("multi_scale_deformable_attention", torch::kCUDA, &multi_scale_deformable_attention);
}

REGISTER_EXTENSION(TORCH_EXTENSION_NAME)
