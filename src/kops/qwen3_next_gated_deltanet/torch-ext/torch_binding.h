#pragma once

#include <torch/torch.h>

at::Tensor qwen3_next_gated_deltanet(at::Tensor q, at::Tensor k, at::Tensor v, at::Tensor g, at::Tensor beta,
                                     int64_t chunk_size);
