# 34 · GDN — Global Dense Normalization (MoE routing)

**Why compile loses.** Qwen 3.5's answer to MoE load-balancing: instead of an auxiliary load-balance loss, **Global Dense Normalization** normalizes activation magnitudes *globally* before the router, letting experts specialize naturally while preventing expert collapse. The op is a global reduction (compute statistics across a large activation set) followed by a normalize+route — a reduction whose scope crosses the usual fusion boundaries, and which feeds directly into the data-dependent MoE routing (#3). `torch.compile` will split the global reduction from the routing; a fused kernel does the normalize-then-top-k in one pass.

**Source.** Write our own; reference from Qwen 3.5 GDN materials.

**Config sketch.** `Config` (non-Hub), dtype bf16. `baseline` = eager GDN: compute global activation stats, normalize, then `topk` router. `custom` = fused normalize + router. Inputs: hidden states `x` (tokens, hidden), router weights, num_experts, top-k. `verify` at atol ~2e-2 on the normalized routing logits / selected experts (pin so selection matches).

**Inputs to think about.** tokens=8192, hidden=4096, experts=128–512, top-k=8. Large expert counts (Qwen 3.5 / DeepSeek-scale) are where routing efficiency matters.

**Difficulty.** Medium — a global reduction fused with top-k routing; pairs directly with issue 35 (fine-grained MoE) as its front-end. Get the global stat scope right (per-tensor vs per-token-group).

**Refs.** Qwen 3.5 GDN: https://qubittool.com/blog/llm-landscape-may-2026-deepseek-qwen-llama-comparison · feeds issue 35.
