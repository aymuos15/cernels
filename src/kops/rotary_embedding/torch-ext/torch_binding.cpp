#include <torch/library.h>

#include "registration.h"
#include "torch_binding.h"

TORCH_LIBRARY_EXPAND(TORCH_EXTENSION_NAME, ops) {
    ops.def("rotary_embedding(Tensor q, Tensor k, Tensor cos, Tensor sin) -> (Tensor, Tensor)");
    ops.impl("rotary_embedding", torch::kCUDA, &rotary_embedding);
}

REGISTER_EXTENSION(TORCH_EXTENSION_NAME)
