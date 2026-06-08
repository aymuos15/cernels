"""qwen3_next_moe_experts loader.

Everything here runs in the timed path; only the raw hidden state + the
Qwen3NextSparseMoeBlock module come from inputs().

  - router (Qwen3NextTopKRouter: linear + fp32 softmax + top-10 + renorm) — same code as
    the baseline (block.gate).
  - CUDA op: token routing + per-expert plain-SwiGLU grouped GEMM (gate_up/down, no bias).
  - shared expert: the always-on gated dense MLP, sigmoid(gate(x)) * mlp(x), added on top.

Qwen3 weight layout: F.linear(x, W) so gate_up_proj[e] is [2F, H] and down_proj[e] is
[H, F]; the CUDA op consumes these directly (it transposes inside the GEMM).
"""

import torch


from kops.registry._local import load


def kernel(x, block):
    if block is None:
        raise RuntimeError("block module not available (meta device)")
    bsz, seqlen, hidden = x.shape
    xf = x.reshape(-1, hidden).contiguous()
    with torch.no_grad():
        # Router: identical to the baseline (block.gate -> logits, scores, indices).
        _logits, scores, indices = block.gate(xf)

        e = block.experts
        routed = load("qwen3_next_moe_experts").qwen3_next_moe_experts(
            xf,
            e.gate_up_proj.data.contiguous(),  # [E, 2F, H]
            e.down_proj.data.contiguous(),  # [E, H, F]
            indices.to(torch.int32).contiguous(),
            scores.float().contiguous(),
            int(block.gate.top_k),
        )

        # Always-on gated shared expert (dense, every token).
        shared = block.shared_expert(xf)
        shared = torch.sigmoid(block.shared_expert_gate(xf)) * shared
        out = routed + shared

    return out.reshape(bsz, seqlen, hidden)
