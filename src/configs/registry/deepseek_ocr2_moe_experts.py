"""DeepseekOcr2TextExperts — DeepSeek-OCR-2 routed grouped expert SwiGLU GEMM.

Reference: the real transformers ``DeepseekOcr2TextExperts`` instantiated from a
``DeepseekOcr2TextConfig`` (deepseek-community/DeepSeek-OCR-2 text shape: hidden 1280,
per-expert intermediate 896, 64 routed experts, top-6) and CALLED — never hand-written
(see docs/guide/setting_up_baselines.md). The eager forward loops over hit experts with
one_hot masks + torch.where — the per-expert-loop pathology the grouped-GEMM kernel
family (gpt_oss_moe_experts, qwen3_next_moe_experts, cohere2_moe_experts) beats.
Constant-swap port of cohere2_moe_experts (.issues/kernel/09).

Routing is NOT part of this op: ``DeepseekOcr2TextExperts`` consumes
(hidden_states_flat, top_k_index, top_k_weights) and the router
(``DeepseekOcr2TextMoe.route_tokens_to_experts``: fp32 softmax + greedy top-6, scaled by
``routed_scaling_factor``) lives outside it, as do the 2 shared experts (added
sequentially after ``self.experts(...)``). inputs() therefore builds the raw routing
tensors the real caller would pass — greedy top-6 over a softmax of seeded random logits
across 64 experts, scores x routed_scaling_factor — pinned by seed so every workload
scores the same selection. The gather, both grouped GEMMs, the SwiGLU, the router-weight
scale and the scatter-add stay in the timed path.
"""

import torch

from configs.base import Config
from kops.registry.deepseek_ocr2_moe_experts import decode_kernel as deepseek_ocr2_moe_decode_kernel
from kops.registry.deepseek_ocr2_moe_experts import kernel as deepseek_ocr2_moe_kernel


class DeepseekOcr2MoeExperts(Config):
    name = "deepseek_ocr2_moe_experts"
    dtype = torch.bfloat16
    op = "transformers DeepseekOcr2TextExperts (grouped expert SwiGLU GEMM, 64 experts top-6)"
    use_compile = True

    _tokens: int = 2048  # b=1, seq 2048 prefill (comparable to the MoE-experts family)
    _hidden: int = 1280
    _ffn: int = 896  # moe_intermediate_size (per-expert)
    _num_experts: int = 64
    _top_k: int = 6
    _routed_scaling_factor: float = 1.0  # deepseek-community/DeepSeek-OCR-2 text_config

    def _build_experts(self, device, dtype):
        from transformers.models.deepseek_ocr2.configuration_deepseek_ocr2 import DeepseekOcr2TextConfig
        from transformers.models.deepseek_ocr2.modeling_deepseek_ocr2 import (
            DeepseekOcr2TextExperts as DeepseekOcr2TextExpertsModule,
        )

        cfg = DeepseekOcr2TextConfig(
            hidden_size=self._hidden,
            moe_intermediate_size=self._ffn,
            n_routed_experts=self._num_experts,
            num_experts_per_tok=self._top_k,
            routed_scaling_factor=self._routed_scaling_factor,
            topk_method="greedy",
            hidden_act="silu",
        )
        torch.manual_seed(0)
        experts = DeepseekOcr2TextExpertsModule(cfg).to(device=device, dtype=dtype)
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
        # Mimic DeepseekOcr2TextMoe.route_tokens_to_experts: fp32 softmax over 64-expert
        # logits, greedy top-6, scores scaled by routed_scaling_factor (NOT sigmoid like
        # cohere2, no top-k renorm).
        router_logits = torch.randn(self._tokens, self._num_experts, device=device)
        scores = router_logits.softmax(dim=-1, dtype=torch.float32)
        top_k_scores, top_k_index = torch.topk(scores, self._top_k, dim=-1)
        top_k_weights = (top_k_scores * self._routed_scaling_factor).to(dtype)
        return x, top_k_index, top_k_weights, experts

    def baseline(self, x, top_k_index, top_k_weights, experts):
        if experts is None:
            raise RuntimeError("experts module not available (meta device)")
        with torch.no_grad():
            return experts(x, top_k_index, top_k_weights)

    custom = staticmethod(deepseek_ocr2_moe_kernel)

    def verify(self, out, ref):
        # Reordered bf16 grouped-GEMM accumulation (reference index_add_ is bf16, the
        # kernel accumulates fp32): combined atol/rtol ~2e-2, as in the MoE family.
        return bool(torch.allclose(out.float(), ref.float(), atol=2e-2, rtol=2e-2))


class DeepseekOcr2MoeExpertsDecode(DeepseekOcr2MoeExperts):
    """Decode-shape twin: n_tokens=1, b=1 single-token decode.

    Same op contract and reference module; at 1 token / top-6 the work is 6 gather-GEMVs
    + SwiGLU + 6 scaled accumulations — memory-bound on the touched expert weights and
    launch-bound in eager. Second config over the same deepseek_ocr2_moe_experts kops
    repo (RULES.md §1: the kernel keeps its one canonical slug; this config only changes
    the shape and routes to the kernel's fused gather-GEMV decode entry point).
    """

    name = "deepseek_ocr2_moe_experts_decode"
    _tokens = 1

    custom = staticmethod(deepseek_ocr2_moe_decode_kernel)
