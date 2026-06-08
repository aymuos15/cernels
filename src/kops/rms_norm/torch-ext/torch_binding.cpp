#include <torch/library.h>

#include "registration.h"
#include "torch_binding.h"

TORCH_LIBRARY_EXPAND(TORCH_EXTENSION_NAME, ops) {
    ops.def("rms_norm(Tensor x, Tensor weight, float eps) -> Tensor");
    ops.impl("rms_norm", torch::kCUDA, &rms_norm);
}

REGISTER_EXTENSION(TORCH_EXTENSION_NAME)
