#pragma once

#include <torch/torch.h>

at::Tensor cohere2_moe_experts(at::Tensor x, at::Tensor gate_up_proj, at::Tensor down_proj, at::Tensor indices,
                               at::Tensor weights, int64_t topk);

at::Tensor cohere2_moe_experts_decode(at::Tensor x, at::Tensor gate_up_proj, at::Tensor down_proj, at::Tensor indices,
                                      at::Tensor weights, int64_t topk);
