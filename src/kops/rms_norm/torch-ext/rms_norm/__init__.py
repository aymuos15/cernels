from ._ops import ops  # type: ignore  # _ops is generated at build time


def rms_norm(x, weight, eps):
    return ops.rms_norm(x, weight, eps)


__all__ = ["rms_norm"]
