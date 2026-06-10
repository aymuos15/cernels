"""Cohere2Moe routed experts — North Mini Code grouped expert SwiGLU GEMM.

Reference: the real transformers ``Cohere2MoeExperts`` instantiated from a
``Cohere2MoeConfig`` (CohereLabs/North-Mini-Code-1.0 shape: hidden 2048, per-expert
intermediate 768, 128 experts, top-8, no shared expert) and CALLED — never hand-written
(see docs/guide/setting_up_baselines.md). The eager forward loops over hit experts with
one_hot masks + torch.where — the per-expert-loop pathology the grouped-GEMM kernel
family (gpt_oss_moe_experts, qwen3_next_moe_experts) beats.

Unlike qwen3_next_moe_experts, routing is NOT part of this op: ``Cohere2MoeExperts``
consumes (hidden_states_flat, top_k_index, top_k_weights) and the router
(``Cohere2MoeTopKRouter``: linear + top-8 + sigmoid, norm_topk_prob=False so no renorm)
lives outside it. inputs() therefore builds the raw routing tensors the real caller
would pass — uniform top-8 indices over 128 experts and sigmoid scores in (0,1), both
pinned by seed so every workload scores the same selection. The gather, both grouped
GEMMs, the SwiGLU, the router-weight scale and the scatter-add stay in the timed path.
"""

import torch

from configs.base import Config
from kops.registry.cohere2_moe_experts import decode_kernel as cohere2_moe_decode_kernel
from kops.registry.cohere2_moe_experts import kernel as cohere2_moe_kernel


class Cohere2MoeExperts(Config):
    name = "cohere2_moe_experts"
    dtype = torch.bfloat16
    op = "transformers Cohere2MoeExperts (grouped expert SwiGLU GEMM, 128 experts top-8)"
    use_compile = True

    _tokens: int = 2048  # b=1, seq 2048 prefill (the model-showcase shape)
    _hidden: int = 2048
    _ffn: int = 768  # per-expert intermediate_size
    _num_experts: int = 128
    _top_k: int = 8

    def _build_experts(self, device, dtype):
        from transformers.models.cohere2_moe.configuration_cohere2_moe import Cohere2MoeConfig
        from transformers.models.cohere2_moe.modeling_cohere2_moe import (
            Cohere2MoeExperts as Cohere2MoeExpertsModule,
        )

        cfg = Cohere2MoeConfig(
            hidden_size=self._hidden,
            intermediate_size=self._ffn,
            num_experts=self._num_experts,
            num_experts_per_tok=self._top_k,
            num_shared_experts=0,
            expert_selection_fn="sigmoid",
            norm_topk_prob=False,
            hidden_act="silu",
        )
        torch.manual_seed(0)
        experts = Cohere2MoeExpertsModule(cfg).to(device=device, dtype=dtype)
        # gate_up_proj / down_proj are torch.empty in __init__; fill all params small.
        with torch.no_grad():
            for p in experts.parameters():
                p.normal_(0.0, 0.02)
        experts.eval()
        return experts

    def inputs(self, device, dtype):
        if str(device) == "meta":
            return (
                torch.empty(self._tokens, self._hidden, device=device, dtype=dtype),
                torch.zeros(self._tokens, self._top_k, device=device, dtype=torch.int64),
                torch.empty(self._tokens, self._top_k, device=device, dtype=dtype),
                None,
            )
        experts = self._build_experts(device, dtype)
        torch.manual_seed(1)  # pin routing -> identical selection across workloads
        x = torch.randn(self._tokens, self._hidden, device=device, dtype=dtype)
        # Mimic Cohere2MoeTopKRouter output: top-8 of random logits (uniform-ish load
        # over 128 experts, distinct per token), sigmoid scores, no top-k renorm.
        router_logits = torch.randn(self._tokens, self._num_experts, device=device)
        top_k_scores, top_k_index = torch.topk(router_logits, self._top_k, dim=-1)
        top_k_weights = torch.sigmoid(top_k_scores).to(dtype)
        return x, top_k_index, top_k_weights, experts

    def baseline(self, x, top_k_index, top_k_weights, experts):
        if experts is None:
            raise RuntimeError("experts module not available (meta device)")
        with torch.no_grad():
            return experts(x, top_k_index, top_k_weights)

    custom = staticmethod(cohere2_moe_kernel)

    def verify(self, out, ref):
        # Reordered bf16 grouped-GEMM accumulation (reference index_add_ is bf16, the
        # kernel accumulates fp32): combined atol/rtol ~2e-2, as in qwen3_next_moe_experts.
        return bool(torch.allclose(out.float(), ref.float(), atol=2e-2, rtol=2e-2))


class Cohere2MoeExpertsDecode(Cohere2MoeExperts):
    """Decode-shape twin (.issues/kernel/08): n_tokens=1, b=1 single-token decode.

    Same op contract and reference module; at 1 token / top-8 the work is 8 gather-GEMVs
    + SwiGLU + 8 scaled accumulations — memory-bound on the touched expert weights and
    launch-bound in eager. Second config over the same cohere2_moe_experts kops repo
    (RULES.md §1: the kernel keeps its one canonical slug; this config only changes the
    shape and routes to the kernel's fused gather-GEMV decode entry point).
    """

    name = "cohere2_moe_experts_decode"
    _tokens = 1

    custom = staticmethod(cohere2_moe_decode_kernel)
