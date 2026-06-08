"""rotary_embedding loader."""

from kops.registry._local import load


def kernel(q, k, cos, sin):
    return load("rotary_embedding").rotary_embedding(q, k, cos, sin)
