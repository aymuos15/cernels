#include <torch/library.h>

#include "registration.h"
#include "torch_binding.h"

TORCH_LIBRARY_EXPAND(TORCH_EXTENSION_NAME, ops) {
    ops.def("rope(Tensor q, Tensor k, Tensor cos, Tensor sin) -> (Tensor, Tensor)");
    ops.impl("rope", torch::kCUDA, &rope);
}

REGISTER_EXTENSION(TORCH_EXTENSION_NAME)
