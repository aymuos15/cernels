#include <torch/library.h>

#include "registration.h"
#include "torch_binding.h"

TORCH_LIBRARY_EXPAND(TORCH_EXTENSION_NAME, ops) {
    ops.def("rope3d(Tensor q, Tensor k, Tensor emb) -> (Tensor, Tensor)");
    ops.impl("rope3d", torch::kCUDA, &rope3d);
}

REGISTER_EXTENSION(TORCH_EXTENSION_NAME)
