#include <torch/library.h>

#include "registration.h"
#include "torch_binding.h"

TORCH_LIBRARY_EXPAND(TORCH_EXTENSION_NAME, ops) {
    ops.def("non_maximum_suppression(Tensor boxes, float thresh) -> Tensor");
    ops.impl("non_maximum_suppression", torch::kCUDA, &non_maximum_suppression);
}

REGISTER_EXTENSION(TORCH_EXTENSION_NAME)
