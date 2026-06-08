"""gpt-oss MoE experts — clamped/limited SwiGLU + grouped expert matmul.

Reference is the real transformers ``GptOssMLP`` CALLED, never hand-written. The gpt-oss
SwiGLU is NOT the canonical silu_mul:
    gate, up = gate_up[..., ::2], gate_up[..., 1::2]   # interleaved (even/odd) split
    gate = gate.clamp(max=7.0); up = up.clamp(-7.0, 7.0)
    glu  = gate * sigmoid(gate * 1.702)
    out  = (up + 1) * glu                              # note the (up + 1) bias term

Timed path (baseline and custom): router linear + top-k + softmax, token routing,
per-expert gate_up/down matmul (with biases), and the clamped-limited SwiGLU.
Precomputed in inputs(): ONLY the raw hidden-state tensor and the GptOssMLP module
(weight storage). The router weight is pinned (seeded, no jitter) so expert selection is
identical across workloads.
"""

import torch

from configs.base import Config
from kops.registry.gpt_oss_moe_experts import kernel as gpt_oss_moe_kernel


class GptOssMoeExperts(Config):
    name = "gpt_oss_moe_experts"
    dtype = torch.bfloat16
    op = "transformers GptOssMLP (clamped-SwiGLU grouped MoE)"
    use_compile = True

    _tokens: int = 4096
    _hidden: int = 2880
    _ffn: int = 2880
    _num_experts: int = 32
    _top_k: int = 4

    def _build_mlp(self, device, dtype):
        from transformers.models.gpt_oss.configuration_gpt_oss import GptOssConfig
        from transformers.models.gpt_oss.modeling_gpt_oss import GptOssMLP

        cfg = GptOssConfig(
            hidden_size=self._hidden,
            intermediate_size=self._ffn,
            num_local_experts=self._num_experts,
            num_experts_per_tok=self._top_k,
        )
        torch.manual_seed(0)  # pin router + expert weights -> deterministic routing
        mlp = GptOssMLP(cfg).to(device=device, dtype=dtype)
        # gate_up/down_proj are torch.empty in __init__ — must fill before use
        with torch.no_grad():
            for p in mlp.parameters():
                p.normal_(0.0, 0.02)
        mlp.eval()
        return mlp

    def inputs(self, device, dtype):
        if str(device) == "meta":
            return torch.empty(1, self._tokens, self._hidden, device=device, dtype=dtype), None
        mlp = self._build_mlp(device, dtype)
        torch.manual_seed(1)
        x = torch.randn(1, self._tokens, self._hidden, device=device, dtype=dtype)
        return x, mlp

    def baseline(self, x, mlp):
        if mlp is None:
            raise RuntimeError("mlp module not available (meta device)")
        with torch.no_grad():
            out, _scores = mlp(x)
        return out

    custom = staticmethod(gpt_oss_moe_kernel)

    def verify(self, out, ref):
        # Reordered bf16 grouped-GEMM accumulation: combined atol/rtol ~2e-2 — the few
        # elements above the bare 2e-2 atol are larger-magnitude outputs covered by rtol
        # (cosine sim 0.99998; see docs/guide/correctness.md).
        return bool(torch.allclose(out.float(), ref.float(), atol=2e-2, rtol=2e-2))
