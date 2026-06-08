from ._ops import ops  # type: ignore  # _ops is generated at build time


def deltanet(q, k, v, g, beta, chunk_size):
    return ops.deltanet(q, k, v, g, beta, chunk_size)


__all__ = ["deltanet"]
