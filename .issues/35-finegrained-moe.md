# 35 · Fine-grained MoE router + grouped GEMM + combine

**Why compile loses.** The universal 2026 architecture: every major model — DeepSeek V4 (256+ experts), Qwen 3.5, gpt-oss (128 experts, top-4), Llama 4 (every layer MoE) — is ultra-sparse MoE activating ~3–4% of params. The full op is: router top-k over many tiny experts → permute/gather tokens by expert → **grouped GEMM** (variable tokens per expert) → un-permute and weighted-combine. This stacks two compile killers: data-dependent routing (#1) and a data-dependent loop count over experts (#3), which makes Dynamo unroll or pad. A fused MoE kernel (à la MegaBlocks/grouped GEMM, but tuned for many small experts) owns the whole path. Highest-leverage kernel in the backlog by adoption — extends issue 06 to the fine-grained, top-k regime.

**Source.** Write our own (grouped GEMM over a permuted token layout), or bind `kernels-community/megablocks` / `scattermoe` as `lib`; reference the gpt-oss / DeepSeek-V4 MoE configs.

**Config sketch.** `Config` (non-Hub), dtype bf16. `baseline` = a loop/dense reference MoE: route once (fixed router for reproducibility), gather per-expert tokens, apply each expert's gated-SwiGLU MLP, scatter-combine by router weight. `custom` = the fused router+grouped-GEMM+combine kernel. Inputs: hidden (tokens, hidden), router weights, stacked expert weights (E, hidden, ffn) for gate/up/down, top-k. `verify` at atol ~2e-2 with pinned routing.

**Inputs to think about.** tokens=4096, hidden=4096, ffn=1536 (small, fine-grained), **E=128–256**, top-k=4–8, plus a shared expert. Many small experts + low top-k is the 2026 regime and the hardest to make efficient.

**Difficulty.** High — routing/permute/grouped-GEMM/combine plus load-balancing skewed expert assignment. Build from issue 06 (MegaBlocks) and issue 23 (ragged GEMM); pair with issue 34 (GDN) as the router front-end and issue 39 (MXFP4) for the quantized variant.

**Refs.** gpt-oss MoE (top-4/128); DeepSeek-V4 / Qwen 3.5 MoE. See issues 06, 23, 34, 39.
