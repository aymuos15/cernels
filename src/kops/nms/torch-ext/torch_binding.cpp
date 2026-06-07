#include <torch/library.h>

#include "registration.h"
#include "torch_binding.h"

TORCH_LIBRARY_EXPAND(TORCH_EXTENSION_NAME, ops) {
    ops.def("nms(Tensor boxes, float thresh) -> Tensor");
    ops.impl("nms", torch::kCUDA, &nms);
}

REGISTER_EXTENSION(TORCH_EXTENSION_NAME)
