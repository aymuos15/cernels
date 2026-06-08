from ._ops import ops  # type: ignore  # _ops is generated at build time


def qwen3_next_gated_deltanet(q, k, v, g, beta, chunk_size):
    return ops.qwen3_next_gated_deltanet(q, k, v, g, beta, chunk_size)


__all__ = ["qwen3_next_gated_deltanet"]
