"""Qwen3-Next fused gated RMSNorm: reference is the REAL transformers
`Qwen3NextRMSNormGated` module (instantiated and called, never hand-written). Custom is our
fused bf16 CUDA kernel (kops/qwen3_next_gated_rmsnorm/).

Verified forward (transformers main / v4.57.0 — norm-BEFORE-gate, PLAIN weight, fp32 reduce):
    h32 = hidden.float(); var = mean(h32^2, -1)
    hn  = h32 * rsqrt(var + eps)
    h   = weight * hn.to(bf16)        # bf16*bf16, NOT (1+weight)
    out = h * F.silu(gate.float())    # promote, silu in fp32
    return out.to(bf16)
NOTE: the issue text guessed gate-then-norm + zero-centered (1+weight); the real module is
norm-then-gate with plain weight. The verified composition above is what is implemented.

Timed path: silu(gate) in fp32, the elementwise multiply, the RMS reduction in fp32, and the
weight scale. Precomputed in inputs(): only raw hidden_states + gate + weight.
"""

import torch

from configs.base import Config
from kops.registry.qwen3_next_gated_rmsnorm import kernel as gated_rmsnorm_kernel


class Qwen3NextGatedRmsnorm(Config):
    name = "qwen3_next_gated_rmsnorm"
    dtype = torch.bfloat16
    op = "Qwen3NextRMSNormGated"
    custom = staticmethod(gated_rmsnorm_kernel)

    # Qwen3-Next-80B-A3B: the gated RMSNorm runs on core_attn_out reshaped to (-1, head_v_dim).
    # head_v_dim=128 is the normalized dim; rows = seq_tokens * linear_num_value_heads (=32).
    _M = 4096 * 32  # 131072 rows  (tokens=4096 * num_v_heads=32)
    _H = 128  # head_v_dim
    _eps = 1e-6
    _module = None

    def inputs(self, device, dtype):
        hidden = torch.randn(self._M, self._H, device=device, dtype=dtype)
        gate = torch.randn(self._M, self._H, device=device, dtype=dtype)
        weight = torch.randn(self._H, device=device, dtype=dtype)
        return hidden, gate, weight, self._eps

    def _ref_module(self, weight, eps):
        # Instantiate the real transformers module once and pin its weight to the shared tensor.
        from transformers.models.qwen3_next.modeling_qwen3_next import Qwen3NextRMSNormGated

        if self._module is None:
            self._module = Qwen3NextRMSNormGated(weight.shape[0], eps=eps).to(weight.device)
        self._module.weight.data = weight
        self._module.variance_epsilon = eps
        return self._module

    def baseline(self, hidden, gate, weight, eps):
        return self._ref_module(weight, eps)(hidden, gate)

    def verify(self, out, ref) -> bool:
        return bool(torch.allclose(out.float(), ref.float(), atol=2e-2, rtol=2e-2))
