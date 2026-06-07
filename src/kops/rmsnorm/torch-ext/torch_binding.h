#pragma once

#include <torch/torch.h>

at::Tensor rmsnorm(at::Tensor x, at::Tensor weight, double eps);
