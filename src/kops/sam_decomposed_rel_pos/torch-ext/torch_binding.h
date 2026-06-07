#pragma once

#include <torch/torch.h>

at::Tensor sam_decomposed_rel_pos(at::Tensor query, at::Tensor Rh, at::Tensor Rw, at::Tensor attn);
