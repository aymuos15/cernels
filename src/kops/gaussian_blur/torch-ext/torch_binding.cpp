#include <torch/library.h>

#include "registration.h"
#include "torch_binding.h"

TORCH_LIBRARY_EXPAND(TORCH_EXTENSION_NAME, ops) {
    ops.def("gblur(Tensor x, Tensor ky, Tensor kx) -> Tensor");
    ops.impl("gblur", torch::kCUDA, &gblur);
}

REGISTER_EXTENSION(TORCH_EXTENSION_NAME)
