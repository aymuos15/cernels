from ._ops import ops  # type: ignore  # _ops is generated at build time


def non_maximum_suppression(boxes, thresh):
    return ops.non_maximum_suppression(boxes, thresh)


__all__ = ["non_maximum_suppression"]
