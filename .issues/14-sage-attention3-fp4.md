# 14 · SageAttention3 (FP4 attention)

**Why compile loses.** The v3 successor to issue 02: attention with the matmuls run in **FP4** on Blackwell tensor cores, ~1,038 TOPS end-to-end on RTX5090. Stacks two compile-proof advantages — flash-style tiling and a quantized (now 4-bit) core matmul that Inductor can't enter. Distinct from issue 02 (INT8/FP8) by the FP4 precision and the hardware requirement.

**Source.** SageAttention3 (paper/repo). Hardware-gated like issue 13 — needs FP4-capable GPU.

**Config sketch.** `Config`, fp16 inputs (quantized internally to FP4). `baseline` = `scaled_dot_product_attention` fp16, causal. `op`/`custom` = SageAttention3 entry. **Guard on FP4-capable arch.** `verify` with a loose relative/cosine bar (~5%), documented — FP4 attention won't pass tight allclose.

**Inputs to think about.** Prefill-shaped, long sequence: batch 1–4, heads 32, head_dim 128, sequence 4k–16k, causal. Speedup grows with S.

**Difficulty.** Medium on the right GPU; blocked otherwise. Reuse the issue-02 harness and swap precision + correctness bar.

**Refs.** SageAttention line (smooth-quantized attention, v1→v3). See issue 02 for the INT8/FP8 predecessor.
