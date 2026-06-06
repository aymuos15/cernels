# 26 · Fused QK-norm + RoPE + SDPA

**Why compile loses.** Modern decoder blocks (Gemma 4, Qwen 3.5, Qwen3) run, per attention layer: RMSNorm on Q, RMSNorm on K (QK-norm, for long-context stability), rotary on Q and K, then SDPA. `torch.compile` launches the norms and rotary as separate elementwise kernels feeding an opaque SDPA call — it can fuse the pointwise norm/rotary together but cannot fuse them *into* the attention. A custom kernel applies QK-norm + RoPE inside the flash-attention prologue (Q/K normalized and rotated in registers before the QK^T tile), saving the round-trips. Near-universal across 2026 models, so one kernel pays off everywhere.

**Source.** Write our own (Triton flash-attention with a norm+rotary prologue), building on the existing RoPE work in `src/kops/`.

**Config sketch.** `Config` (non-Hub), dtype bf16. `baseline` = the eager sequence: `q = rope(rmsnorm(q)*wq); k = rope(rmsnorm(k)*wk); sdpa(q,k,v)` matching Gemma/Qwen. `custom` = the fused kernel. Inputs: q/k/v (b, h, s, d), per-head RMSNorm weights for q and k, cos/sin, causal flag. `verify` at atol ~2e-2; do the norm reduction in fp32 inside the kernel.

**Inputs to think about.** GQA shapes: b=2, q-heads 32 / kv-heads 8, head_dim 128, sequence 4k–16k. Try with and without QK-norm to isolate the fusion benefit.

**Difficulty.** Medium-high — it's a flash-attention kernel plus a prologue; reuse issue-09 (RMSNorm) and the rotary kernel as building blocks. Pairs with issue 27 (add the sliding-window mask) and issue 26 is the dense-global counterpart.

**Refs.** Gemma 3/4 QK-norm (zero-centered RMSNorm); Qwen3 q/k norm. See issues 09 (RMSNorm), rotary.
