#include <torch/library.h>

#include "registration.h"
#include "torch_binding.h"

TORCH_LIBRARY_EXPAND(TORCH_EXTENSION_NAME, ops) {
    ops.def("deltanet(Tensor q, Tensor k, Tensor v, Tensor g, Tensor beta, int chunk_size) -> Tensor");
    ops.impl("deltanet", torch::kCUDA, &deltanet);
}

REGISTER_EXTENSION(TORCH_EXTENSION_NAME)
