#pragma once

#include <torch/torch.h>

at::Tensor megablocks_moe(at::Tensor x, at::Tensor w1, at::Tensor w2, at::Tensor indices, at::Tensor weights,
                          int64_t topk, int64_t act_id);
