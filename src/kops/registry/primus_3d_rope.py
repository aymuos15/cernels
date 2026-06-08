"""primus_3d_rope loader."""

from kops.registry._local import load


def kernel(q, k, emb):
    return load("primus_3d_rope").primus_3d_rope(q, k, emb)
