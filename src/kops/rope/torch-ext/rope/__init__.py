from ._ops import ops  # type: ignore  # _ops is generated at build time


def rope(q, k, cos, sin):
    return ops.rope(q, k, cos, sin)


__all__ = ["rope"]
