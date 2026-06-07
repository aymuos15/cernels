#include <torch/library.h>

#include "registration.h"
#include "torch_binding.h"

TORCH_LIBRARY_EXPAND(TORCH_EXTENSION_NAME, ops) {
    ops.def("rmsnorm(Tensor x, Tensor weight, float eps) -> Tensor");
    ops.impl("rmsnorm", torch::kCUDA, &rmsnorm);
}

REGISTER_EXTENSION(TORCH_EXTENSION_NAME)
