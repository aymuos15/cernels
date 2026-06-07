#pragma once

#include <torch/torch.h>

at::Tensor silu_mul(at::Tensor gate, at::Tensor up);
