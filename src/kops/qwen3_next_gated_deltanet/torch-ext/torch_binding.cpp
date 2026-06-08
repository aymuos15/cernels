#include <torch/library.h>

#include "registration.h"
#include "torch_binding.h"

TORCH_LIBRARY_EXPAND(TORCH_EXTENSION_NAME, ops) {
    ops.def("qwen3_next_gated_deltanet(Tensor q, Tensor k, Tensor v, Tensor g, Tensor beta, int chunk_size) -> Tensor");
    ops.impl("qwen3_next_gated_deltanet", torch::kCUDA, &qwen3_next_gated_deltanet);
}

REGISTER_EXTENSION(TORCH_EXTENSION_NAME)
