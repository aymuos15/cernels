#include <torch/library.h>

#include "registration.h"
#include "torch_binding.h"

TORCH_LIBRARY_EXPAND(TORCH_EXTENSION_NAME, ops) {
    ops.def("megablocks_moe(Tensor x, Tensor w1, Tensor w2, Tensor indices, Tensor weights, int topk, int act_id) -> "
            "Tensor");
    ops.impl("megablocks_moe", torch::kCUDA, &megablocks_moe);
}

REGISTER_EXTENSION(TORCH_EXTENSION_NAME)
