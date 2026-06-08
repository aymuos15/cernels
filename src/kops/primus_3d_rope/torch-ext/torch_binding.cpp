#include <torch/library.h>

#include "registration.h"
#include "torch_binding.h"

TORCH_LIBRARY_EXPAND(TORCH_EXTENSION_NAME, ops) {
    ops.def("primus_3d_rope(Tensor q, Tensor k, Tensor emb) -> (Tensor, Tensor)");
    ops.impl("primus_3d_rope", torch::kCUDA, &primus_3d_rope);
}

REGISTER_EXTENSION(TORCH_EXTENSION_NAME)
