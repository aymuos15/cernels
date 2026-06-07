#pragma once

#include <torch/torch.h>

at::Tensor nms(at::Tensor boxes, double thresh);
