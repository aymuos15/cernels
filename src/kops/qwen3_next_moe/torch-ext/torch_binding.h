#pragma once

#include <torch/torch.h>

at::Tensor qwen3_next_moe(at::Tensor x, at::Tensor gate_up_proj, at::Tensor down_proj, at::Tensor indices,
                          at::Tensor weights, int64_t topk);
