"""Qwen3-Next high-sparsity MoE experts — plain-SwiGLU grouped MoE + gated shared expert.

Reference: the real transformers ``Qwen3NextSparseMoeBlock`` (``Qwen3NextTopKRouter`` +
``Qwen3NextExperts`` + a gated dense shared expert) instantiated from a qwen3_next config
and CALLED — never hand-written (see docs/guide/setting_up_baselines.md). Ultra-sparse:
512 routed experts, top-10, plus 1 always-on shared expert.

Fast-follow on the shipped ``gpt_oss_moe`` (same fused cuBLAS grouped-GEMM family). The
ONLY deltas vs gpt-oss (all verified against modeling_qwen3_next.py):
  1. PLAIN SwiGLU: routed MLP is ``silu(gate) * up`` with a chunked (gate=first F, up=
     second F) split — NO clamp(+/-7), NO alpha=1.702, NO (up+1) term, NO biases.
  2. Router ``Qwen3NextTopKRouter``: softmax(fp32) over the 512 logits THEN top-10, then
     (norm_topk_prob=True) renormalize the top-10 probs to sum to 1. Returns (logits,
     scores, indices).
  3. Shared expert: an always-on dense Qwen3NextMLP gated by sigmoid:
     ``out += sigmoid(shared_expert_gate(x)) * shared_expert_mlp(x)`` — runs every token
     and is part of the timed op.
Weight layout also differs: Qwen3 uses ``F.linear(x, W)`` so per expert gate_up_proj is
[2F, H] and down_proj is [H, F] (gpt-oss stored the transposes and added biases).

Timed path (BOTH baseline and custom): router linear + softmax + top-k + renorm, token
routing, per-expert gate_up/down GEMMs, the plain SiLU SwiGLU, the gated shared-expert
MLP, and the scatter-combine. Precomputed in inputs(): ONLY the raw hidden-state tensor
and the Qwen3NextSparseMoeBlock module (weight storage) — never the routed/gathered
tokens or activations. The router weight is pinned (seeded, no jitter) so expert
selection is identical across op_eager/op_compile/custom.
"""

import torch

from configs.base import Config
from kops.registry.qwen3_next_moe_experts import kernel as qwen3_next_moe_kernel


class Qwen3NextMoeExperts(Config):
    name = "qwen3_next_moe_experts"
    dtype = torch.bfloat16
    op = "transformers Qwen3NextSparseMoeBlock (plain-SwiGLU grouped MoE + gated shared expert)"
    use_compile = True

    _tokens: int = 4096
    _hidden: int = 2048
    _ffn: int = 512  # moe_intermediate_size
    _shared_ffn: int = 512  # shared_expert_intermediate_size
    _num_experts: int = 512
    _top_k: int = 10

    def _build_block(self, device, dtype):
        from transformers.models.qwen3_next.configuration_qwen3_next import Qwen3NextConfig
        from transformers.models.qwen3_next.modeling_qwen3_next import Qwen3NextSparseMoeBlock

        cfg = Qwen3NextConfig(
            hidden_size=self._hidden,
            moe_intermediate_size=self._ffn,
            shared_expert_intermediate_size=self._shared_ffn,
            num_experts=self._num_experts,
            num_experts_per_tok=self._top_k,
            norm_topk_prob=True,
        )
        torch.manual_seed(0)  # pin router + expert weights -> deterministic routing
        block = Qwen3NextSparseMoeBlock(cfg).to(device=device, dtype=dtype)
        # gate_up_proj / down_proj are torch.empty in __init__; fill all params small.
        with torch.no_grad():
            for p in block.parameters():
                p.normal_(0.0, 0.02)
        block.eval()
        return block

    def inputs(self, device, dtype):
        if str(device) == "meta":
            return torch.empty(1, self._tokens, self._hidden, device=device, dtype=dtype), None
        block = self._build_block(device, dtype)
        torch.manual_seed(1)
        x = torch.randn(1, self._tokens, self._hidden, device=device, dtype=dtype)
        return x, block

    def baseline(self, x, block):
        if block is None:
            raise RuntimeError("block module not available (meta device)")
        with torch.no_grad():
            out = block(x)
        return out

    custom = staticmethod(qwen3_next_moe_kernel)

    def verify(self, out, ref):
        # Reordered bf16 grouped-GEMM accumulation: combined atol/rtol ~2e-2, as in
        # gpt_oss_moe (see docs/guide/correctness.md).
        return bool(torch.allclose(out.float(), ref.float(), atol=2e-2, rtol=2e-2))
