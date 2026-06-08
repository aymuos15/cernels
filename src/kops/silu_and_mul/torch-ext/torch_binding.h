#pragma once

#include <torch/torch.h>

at::Tensor silu_and_mul(at::Tensor gate, at::Tensor up);
