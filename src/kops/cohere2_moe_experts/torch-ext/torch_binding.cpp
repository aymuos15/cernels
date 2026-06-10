#include <torch/library.h>

#include "registration.h"
#include "torch_binding.h"

TORCH_LIBRARY_EXPAND(TORCH_EXTENSION_NAME, ops) {
    ops.def("cohere2_moe_experts(Tensor x, Tensor gate_up_proj, Tensor down_proj, Tensor indices, Tensor weights, "
            "int topk) -> Tensor");
    ops.impl("cohere2_moe_experts", torch::kCUDA, &cohere2_moe_experts);
    ops.def("cohere2_moe_experts_decode(Tensor x, Tensor gate_up_proj, Tensor down_proj, Tensor indices, "
            "Tensor weights, int topk) -> Tensor");
    ops.impl("cohere2_moe_experts_decode", torch::kCUDA, &cohere2_moe_experts_decode);
}

REGISTER_EXTENSION(TORCH_EXTENSION_NAME)
