#pragma once

#include <torch/torch.h>

at::Tensor gblur(at::Tensor x, at::Tensor ky, at::Tensor kx);
