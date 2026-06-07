#include <torch/library.h>

#include "registration.h"
#include "torch_binding.h"

TORCH_LIBRARY_EXPAND(TORCH_EXTENSION_NAME, ops) {
    ops.def("silu_mul(Tensor gate, Tensor up) -> Tensor");
    ops.impl("silu_mul", torch::kCUDA, &silu_mul);
}

REGISTER_EXTENSION(TORCH_EXTENSION_NAME)
