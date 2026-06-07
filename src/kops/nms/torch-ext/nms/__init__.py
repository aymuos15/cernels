from ._ops import ops  # type: ignore  # _ops is generated at build time


def nms(boxes, thresh):
    return ops.nms(boxes, thresh)


__all__ = ["nms"]
