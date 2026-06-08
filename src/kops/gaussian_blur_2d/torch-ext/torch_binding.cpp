#include <torch/library.h>

#include "registration.h"
#include "torch_binding.h"

TORCH_LIBRARY_EXPAND(TORCH_EXTENSION_NAME, ops) {
    ops.def("gaussian_blur_2d(Tensor x, Tensor ky, Tensor kx) -> Tensor");
    ops.impl("gaussian_blur_2d", torch::kCUDA, &gaussian_blur_2d);
}

REGISTER_EXTENSION(TORCH_EXTENSION_NAME)
