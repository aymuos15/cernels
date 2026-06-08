from ._ops import ops  # type: ignore  # _ops is generated at build time


def gated_rmsnorm(x, gate, weight, eps):
    return ops.gated_rmsnorm(x, gate, weight, eps)


__all__ = ["gated_rmsnorm"]
