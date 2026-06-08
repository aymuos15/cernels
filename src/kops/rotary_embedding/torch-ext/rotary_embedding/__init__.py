from ._ops import ops  # type: ignore  # _ops is generated at build time


def rotary_embedding(q, k, cos, sin):
    return ops.rotary_embedding(q, k, cos, sin)


__all__ = ["rotary_embedding"]
