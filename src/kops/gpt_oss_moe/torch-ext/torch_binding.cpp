#include <torch/library.h>

#include "registration.h"
#include "torch_binding.h"

TORCH_LIBRARY_EXPAND(TORCH_EXTENSION_NAME, ops) {
    ops.def("gpt_oss_moe(Tensor x, Tensor gate_up_proj, Tensor gate_up_bias, Tensor down_proj, Tensor down_bias, "
            "Tensor indices, Tensor weights, int topk) -> Tensor");
    ops.impl("gpt_oss_moe", torch::kCUDA, &gpt_oss_moe);
}

REGISTER_EXTENSION(TORCH_EXTENSION_NAME)
