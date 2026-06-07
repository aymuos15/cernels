#include <torch/library.h>

#include "registration.h"
#include "torch_binding.h"

TORCH_LIBRARY_EXPAND(TORCH_EXTENSION_NAME, ops) {
    ops.def("moe_grouped_gemm(Tensor x, Tensor w1, Tensor w2, Tensor indices, Tensor weights, int topk, int act_id) -> "
            "Tensor");
    ops.impl("moe_grouped_gemm", torch::kCUDA, &moe_grouped_gemm);
}

REGISTER_EXTENSION(TORCH_EXTENSION_NAME)
