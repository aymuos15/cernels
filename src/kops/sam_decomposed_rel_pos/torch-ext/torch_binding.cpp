#include <torch/library.h>

#include "registration.h"
#include "torch_binding.h"

TORCH_LIBRARY_EXPAND(TORCH_EXTENSION_NAME, ops) {
    ops.def("sam_decomposed_rel_pos(Tensor query, Tensor Rh, Tensor Rw, Tensor attn) -> Tensor");
    ops.impl("sam_decomposed_rel_pos", torch::kCUDA, &sam_decomposed_rel_pos);
}

REGISTER_EXTENSION(TORCH_EXTENSION_NAME)
