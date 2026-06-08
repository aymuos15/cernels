from ._ops import ops  # type: ignore  # _ops is generated at build time


def gaussian_blur_2d(x, ky, kx):
    return ops.gaussian_blur_2d(x, ky, kx)


__all__ = ["gaussian_blur_2d"]
