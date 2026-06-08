#pragma once

#include <torch/torch.h>

std::tuple<at::Tensor, at::Tensor> rotary_embedding(at::Tensor q, at::Tensor k, at::Tensor cos, at::Tensor sin);
