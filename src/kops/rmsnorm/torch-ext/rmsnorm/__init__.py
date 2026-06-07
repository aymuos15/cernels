from ._ops import ops  # type: ignore  # _ops is generated at build time


def rmsnorm(x, weight, eps):
    return ops.rmsnorm(x, weight, eps)


__all__ = ["rmsnorm"]
