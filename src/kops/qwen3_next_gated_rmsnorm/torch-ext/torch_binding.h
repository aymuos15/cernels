#pragma once

#include <torch/torch.h>

at::Tensor gated_rmsnorm(at::Tensor x, at::Tensor gate, at::Tensor weight, double eps);
