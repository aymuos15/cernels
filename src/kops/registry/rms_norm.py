"""rms_norm loader — matches F.rms_norm (bf16)."""

from kops.registry._local import load


def kernel(x, weight, eps):
    return load("rms_norm").rms_norm(x, weight, float(eps))
