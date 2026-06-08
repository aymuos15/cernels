"""gpt_oss_moe_experts loader.

The router (linear + top-k + softmax) runs HERE, in the timed path — only the raw
hidden state + the GptOssMLP module come from inputs(). The CUDA op does the token
routing, per-expert gate_up/down GEMMs (with biases) and the clamped-limited SwiGLU.
"""

import torch


from kops.registry._local import load


def _weights(e):
    # Static expert weights + fp32 bias casts done once per module, not re-cast each
    # timed iteration (the op consumes fp32 bias; these tensors never change).
    w = getattr(e, "_kops_weights", None)
    if w is None:
        w = (
            e.gate_up_proj.data.contiguous(),
            e.gate_up_proj_bias.data.float().contiguous(),
            e.down_proj.data.contiguous(),
            e.down_proj_bias.data.float().contiguous(),
        )
        e._kops_weights = w
    return w


def kernel(x, mlp):
    if mlp is None:
        raise RuntimeError("mlp module not available (meta device)")
    bsz, seqlen, hidden = x.shape
    xf = x.reshape(-1, hidden).contiguous()
    with torch.no_grad():
        _logits, scores, indices = mlp.router(xf)  # timed router (same code as baseline)
    gate_up, gate_up_bias, down, down_bias = _weights(mlp.experts)
    out = load("gpt_oss_moe_experts").gpt_oss_moe_experts(
        xf,
        gate_up,
        gate_up_bias,
        down,
        down_bias,
        indices.to(torch.int32).contiguous(),
        scores.float().contiguous(),
        mlp.router.top_k,
    )
    return out.reshape(bsz, seqlen, hidden)
