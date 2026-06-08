#pragma once

#include <torch/torch.h>

at::Tensor gaussian_blur_2d(at::Tensor x, at::Tensor ky, at::Tensor kx);
