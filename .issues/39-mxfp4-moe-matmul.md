# 39 · MXFP4 MoE matmul

**Why compile loses.** gpt-oss stores ~90% of its parameters (the MoE expert weights) in **MXFP4** (4.25 bits/param), and DeepSeek V4-Flash and others follow. This is issue 13's FP4 microscaling GEMM specialized to the grouped-expert layout of issue 35: per-expert FP4 weight blocks with shared exponents, grouped/variable token counts, run on Blackwell tensor cores. `torch.compile` can enter neither the quantized matmul (#4) nor the data-dependent expert grouping (#3), so the entire hot path of these models is opaque to it. Combines the two biggest structural wins — FP4 compute and MoE routing — into one kernel. **Blackwell/FP4-gated.**

**Source.** Write our own (grouped FP4 GEMM, CUTLASS/QuTLASS) or bind a 4-bit MoE kernel as `lib`; reference gpt-oss MXFP4 MoE.

**Config sketch.** `Config` (non-Hub), bf16 activations. `baseline` = dequantize expert weights to bf16 and run the issue-35 reference MoE (route → grouped GEMM → combine). `custom` = grouped MXFP4 GEMM with per-block scales over the permuted token layout. Inputs: hidden (tokens, hidden), router output, expert weights pre-quantized to MXFP4 (E, ffn, hidden) + per-32 block scales, top-k. **Guard on FP4-capable arch** (cc ≥ 10.0). `verify` with a relative/cosine bar (~3%), documented.

**Inputs to think about.** tokens=4096, hidden=4096, ffn=1536, E=128, top-k=4 (gpt-oss-shaped). The regime where FP4 + sparsity compounds.

**Difficulty.** High — issue 13 (FP4 GEMM) × issue 35 (grouped MoE), plus hardware gating. Land 13 and 35 independently first, then fuse.

**Refs.** gpt-oss MXFP4: https://www.centron.de/en/tutorial/openai-gpt-oss-explained-architecture-mxfp4-quantization-120b-20b-models/ · QuTLASS FP4. See issues 13, 35.
