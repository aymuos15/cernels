"""Qwen3-Next Gated DeltaNet loader.

Returns core_attn_out (b, seq, num_heads, head_dim), matching torch_chunk_gated_delta_rule
(use_qk_l2norm_in_kernel=True, scale=1/sqrt(head_dim)). The l2norm, cumulative gate decay,
beta write-strength and the fp32 cross-chunk state update all run inside the fused CUDA op.
"""

from kops.registry._local import load


def kernel(q, k, v, g, beta, chunk_size):
    return load("qwen3_next_gated_deltanet").qwen3_next_gated_deltanet(q, k, v, g, beta, int(chunk_size))
