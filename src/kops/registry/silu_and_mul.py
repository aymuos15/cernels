"""silu_and_mul loader."""

from kops.registry._local import load


def kernel(gate, up):
    return load("silu_and_mul").silu_and_mul(gate, up)
