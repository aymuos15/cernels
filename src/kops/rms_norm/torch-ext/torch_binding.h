#pragma once

#include <torch/torch.h>

at::Tensor rms_norm(at::Tensor x, at::Tensor weight, double eps);
