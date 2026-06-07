#include <torch/library.h>

#include "registration.h"
#include "torch_binding.h"

TORCH_LIBRARY_EXPAND(TORCH_EXTENSION_NAME, ops) {
    ops.def("roi_align(Tensor input, Tensor boxes, int output_size, float spatial_scale, int sampling_ratio, bool "
            "aligned) -> Tensor");
    ops.impl("roi_align", torch::kCUDA, &roi_align);
}

REGISTER_EXTENSION(TORCH_EXTENSION_NAME)
