#pragma once

#include <torch/torch.h>

at::Tensor deltanet(at::Tensor q, at::Tensor k, at::Tensor v, at::Tensor g, at::Tensor beta, int64_t chunk_size);
