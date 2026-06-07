from ._ops import ops  # type: ignore  # _ops is generated at build time


def gblur(x, ky, kx):
    return ops.gblur(x, ky, kx)


__all__ = ["gblur"]
