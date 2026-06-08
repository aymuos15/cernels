#pragma once

#include <torch/torch.h>

at::Tensor non_maximum_suppression(at::Tensor boxes, double thresh);
