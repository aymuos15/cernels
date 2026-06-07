from ._ops import ops  # type: ignore  # _ops is generated at build time


def moe_grouped_gemm(x, w1, w2, indices, weights, topk, act_id):
    return ops.moe_grouped_gemm(x, w1, w2, indices, weights, topk, act_id)


__all__ = ["moe_grouped_gemm"]
