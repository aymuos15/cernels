from ._ops import ops  # type: ignore  # _ops is generated at build time


def megablocks_moe(x, w1, w2, indices, weights, topk, act_id):
    return ops.megablocks_moe(x, w1, w2, indices, weights, topk, act_id)


__all__ = ["megablocks_moe"]
