from ._ops import ops  # type: ignore  # _ops is generated at build time


def silu_and_mul(gate, up):
    return ops.silu_and_mul(gate, up)


__all__ = ["silu_and_mul"]
