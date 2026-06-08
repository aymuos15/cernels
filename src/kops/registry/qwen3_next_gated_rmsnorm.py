"""qwen3_next_gated_rmsnorm loader.

Fused gated RMSNorm (bf16), matching transformers Qwen3NextRMSNormGated.forward
(norm-before-gate, plain weight).
"""

from kops.registry._local import load


def kernel(hidden_states, gate, weight, eps):
    return load("qwen3_next_gated_rmsnorm").qwen3_next_gated_rmsnorm(hidden_states, gate, weight, float(eps))
