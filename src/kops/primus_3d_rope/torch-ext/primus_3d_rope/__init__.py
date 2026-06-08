from ._ops import ops  # type: ignore  # _ops is generated at build time


def primus_3d_rope(q, k, emb):
    return ops.primus_3d_rope(q, k, emb)


__all__ = ["primus_3d_rope"]
