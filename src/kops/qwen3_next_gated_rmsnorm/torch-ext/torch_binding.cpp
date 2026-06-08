#include <torch/library.h>

#include "registration.h"
#include "torch_binding.h"

TORCH_LIBRARY_EXPAND(TORCH_EXTENSION_NAME, ops) {
    ops.def("qwen3_next_gated_rmsnorm(Tensor x, Tensor gate, Tensor weight, float eps) -> Tensor");
    ops.impl("qwen3_next_gated_rmsnorm", torch::kCUDA, &qwen3_next_gated_rmsnorm);
}

REGISTER_EXTENSION(TORCH_EXTENSION_NAME)
