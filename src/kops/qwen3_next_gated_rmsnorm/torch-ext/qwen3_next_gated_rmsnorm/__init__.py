from ._ops import ops  # type: ignore  # _ops is generated at build time


def qwen3_next_gated_rmsnorm(x, gate, weight, eps):
    return ops.qwen3_next_gated_rmsnorm(x, gate, weight, eps)


__all__ = ["qwen3_next_gated_rmsnorm"]
