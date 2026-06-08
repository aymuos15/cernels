#pragma once

#include <torch/torch.h>

at::Tensor gpt_oss_moe(at::Tensor x, at::Tensor gate_up_proj, at::Tensor gate_up_bias, at::Tensor down_proj,
                       at::Tensor down_bias, at::Tensor indices, at::Tensor weights, int64_t topk);
