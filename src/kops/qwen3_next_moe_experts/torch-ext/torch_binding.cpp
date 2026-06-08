#include <torch/library.h>

#include "registration.h"
#include "torch_binding.h"

TORCH_LIBRARY_EXPAND(TORCH_EXTENSION_NAME, ops) {
    ops.def("qwen3_next_moe_experts(Tensor x, Tensor gate_up_proj, Tensor down_proj, Tensor indices, Tensor weights, "
            "int topk) -> Tensor");
    ops.impl("qwen3_next_moe_experts", torch::kCUDA, &qwen3_next_moe_experts);
}

REGISTER_EXTENSION(TORCH_EXTENSION_NAME)
