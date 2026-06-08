from ._ops import ops  # type: ignore  # _ops is generated at build time


def gpt_oss_moe(x, gate_up_proj, gate_up_bias, down_proj, down_bias, indices, weights, topk):
    return ops.gpt_oss_moe(x, gate_up_proj, gate_up_bias, down_proj, down_bias, indices, weights, topk)


__all__ = ["gpt_oss_moe"]
