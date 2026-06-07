#pragma once

#include <torch/torch.h>

std::tuple<at::Tensor, at::Tensor> rope3d(at::Tensor q, at::Tensor k, at::Tensor emb);
