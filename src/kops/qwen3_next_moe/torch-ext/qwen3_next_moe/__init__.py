from ._ops import ops  # type: ignore  # _ops is generated at build time


def qwen3_next_moe(x, gate_up_proj, down_proj, indices, weights, topk):
    return ops.qwen3_next_moe(x, gate_up_proj, down_proj, indices, weights, topk)


__all__ = ["qwen3_next_moe"]
