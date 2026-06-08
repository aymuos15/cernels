#pragma once

#include <torch/torch.h>

std::tuple<at::Tensor, at::Tensor> primus_3d_rope(at::Tensor q, at::Tensor k, at::Tensor emb);
