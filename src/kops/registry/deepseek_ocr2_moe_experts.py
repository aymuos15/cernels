"""deepseek_ocr2_moe_experts loader.

Receives exactly what the baseline DeepseekOcr2TextExperts.forward receives — flat
hidden states + top_k_index/top_k_weights (routing, softmax-greedy top-6 scaled by
routed_scaling_factor, lives in DeepseekOcr2TextMoe.route_tokens_to_experts, outside
the op; the 2 shared experts also stay outside) — plus the experts module for weight
storage. The gather, both grouped GEMMs, the SwiGLU, the router-weight scale and the
scatter-add all run inside the CUDA op, in the timed path. Weight layout: F.linear(x, W)
so gate_up_proj[e] is [2F, H] and down_proj[e] is [H, F]; the CUDA op consumes these
directly (transposed in the GEMM).
"""

import torch


from kops.registry._local import load


def kernel(x, top_k_index, top_k_weights, experts):
    if experts is None:
        raise RuntimeError("experts module not available (meta device)")
    with torch.no_grad():
        return load("deepseek_ocr2_moe_experts").deepseek_ocr2_moe_experts(
            x.contiguous(),
            experts.gate_up_proj.data.contiguous(),  # [E, 2F, H]
            experts.down_proj.data.contiguous(),  # [E, H, F]
            top_k_index.to(torch.int32).contiguous(),
            top_k_weights.float().contiguous(),
            int(top_k_index.size(1)),
        )


def decode_kernel(x, top_k_index, top_k_weights, experts):
    """Fused gather-GEMV entry point for decode (n_tokens ~ 1..4) — same contract,
    no per-expert cublas loop and no D2H sync (see .issues/kernel/09)."""
    if experts is None:
        raise RuntimeError("experts module not available (meta device)")
    with torch.no_grad():
        return load("deepseek_ocr2_moe_experts").deepseek_ocr2_moe_experts_decode(
            x.contiguous(),
            experts.gate_up_proj.data.contiguous(),  # [E, 2F, H]
            experts.down_proj.data.contiguous(),  # [E, H, F]
            top_k_index.to(torch.int32).contiguous(),
            top_k_weights.float().contiguous(),
            int(top_k_index.size(1)),
        )
