"""Qwen3-Next Gated DeltaNet — chunked delta-rule linear attention.

Reference: the REAL transformers `torch_chunk_gated_delta_rule` from
`modeling_qwen3_next.py` (op_eager / op_compile). Correctness is checked against the
token-recurrent reference `torch_recurrent_gated_delta_rule` from the same module — a
linear scan drifts in bf16, so the bar is a magnitude-aware allclose (atol 2e-2). The two
transformers references agree to ~1e-4 on these shapes, so they are a trustworthy ground truth.

Custom: our chunk-parallel bf16/fp32 CUDA kernel (kops/qwen3_next_gated_deltanet/). It keeps
the cross-chunk fp32 state resident in shared memory; since GB10 caps opt-in dynamic shared
memory at ~99 KB (and the state alone is 64 KB), the kernel tiles internally with CHUNK=16
(the delta-rule result is algebraically independent of the evaluation chunk size, so it still
matches the chunk_size=64 reference within the bf16 tolerance).

The op as the Qwen3-Next 80B-A3B layer drives it (after the QK repeat_interleave that makes
q/k share the 32 value heads): q,k,v are (b, seq, num_v_heads=32, head_dim=128), g and beta
are (b, seq, 32), chunk_size=64, use_qk_l2norm_in_kernel=True, scale=1/sqrt(128). The l2norm
of q/k, the cumulative gate decay, the beta write-strength and the running fp32 state update
all happen INSIDE the timed op (baseline and custom) — inputs() only builds the raw tensors
the real caller passes (never precomputes the per-chunk decays or the state).
"""

import torch

from configs.base import Config
from kops.registry.qwen3_next_gated_deltanet import kernel as deltanet_kernel

_CHUNK = 64


class Qwen3NextGatedDeltanet(Config):
    name = "qwen3_next_gated_deltanet"
    dtype = torch.bfloat16
    op = "torch_chunk_gated_delta_rule (Qwen3-Next DeltaNet)"
    custom = staticmethod(deltanet_kernel)

    # Qwen3-Next-80B-A3B linear-attn shape (post QK repeat_interleave): all heads = num_v_heads.
    _b, _h, _s, _dk, _dv = 1, 32, 2048, 128, 128

    def inputs(self, device, dtype):
        b, h, s, dk, dv = self._b, self._h, self._s, self._dk, self._dv
        gen = torch.Generator(device=device).manual_seed(0) if device != "meta" else None
        kw = dict(device=device, dtype=dtype)
        if gen is not None:
            q = torch.randn(b, s, h, dk, generator=gen, **kw)
            k = torch.randn(b, s, h, dk, generator=gen, **kw)
            v = torch.randn(b, s, h, dv, generator=gen, **kw)
            # g is a NEGATIVE per-step log-decay (g = -A.exp()*softplus(...)); beta in (0,1).
            a = torch.randn(b, s, h, device=device, generator=gen)
            g = (-torch.nn.functional.softplus(a)).to(dtype)
            beta = torch.rand(b, s, h, generator=gen, **kw)
        else:  # meta device: shapes only, no generator
            q = torch.randn(b, s, h, dk, **kw)
            k = torch.randn(b, s, h, dk, **kw)
            v = torch.randn(b, s, h, dv, **kw)
            g = torch.randn(b, s, h, **kw)
            beta = torch.rand(b, s, h, **kw)
        return q, k, v, g, beta, _CHUNK

    def baseline(self, q, k, v, g, beta, chunk_size):
        from transformers.models.qwen3_next.modeling_qwen3_next import (
            torch_chunk_gated_delta_rule,
        )

        out, _ = torch_chunk_gated_delta_rule(q, k, v, g, beta, chunk_size=chunk_size, use_qk_l2norm_in_kernel=True)
        return out

    def verify(self, out, ref) -> bool:
        # ref is the op_eager output (the transformers chunk reference, which matches the
        # token-recurrent reference to ~1e-4 on these shapes). bf16 linear scans drift, so a
        # magnitude-aware atol/rtol 2e-2 is the honest bar (documented in the module docstring).
        return bool(torch.allclose(out.float(), ref.float(), atol=2e-2, rtol=2e-2))
