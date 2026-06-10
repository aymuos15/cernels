from ._ops import ops  # type: ignore  # _ops is generated at build time


def cohere2_moe_experts(x, gate_up_proj, down_proj, indices, weights, topk):
    return ops.cohere2_moe_experts(x, gate_up_proj, down_proj, indices, weights, topk)


def cohere2_moe_experts_decode(x, gate_up_proj, down_proj, indices, weights, topk):
    return ops.cohere2_moe_experts_decode(x, gate_up_proj, down_proj, indices, weights, topk)


__all__ = ["cohere2_moe_experts", "cohere2_moe_experts_decode"]
