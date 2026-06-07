from ._ops import ops  # type: ignore  # _ops is generated at build time


def rope3d(q, k, emb):
    return ops.rope3d(q, k, emb)


__all__ = ["rope3d"]
