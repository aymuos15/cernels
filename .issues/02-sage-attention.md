# 02 · SageAttention (quantized attention)

**Why compile loses.** SageAttention quantizes Q/K (and optionally P/V) to INT8/FP8 and runs the attention matmuls in low precision with per-block smoothing/scaling. `torch.compile` cannot enter a quantized matmul — it can fuse the surrounding dequant pointwise ops but the compute-bound quant GEMM is a black box to it. So this stacks two compile-proof advantages: attention tiling and quantized compute.

**Source.** `kernels-community/sage-attention`.

**Config sketch.** `HubConfig`, dtype fp16 inputs (the kernel quantizes internally). `baseline` is `scaled_dot_product_attention` in fp16. `op` is the kernel's attention entry. Not `out_arg`. `verify` needs a much looser tolerance than usual — INT8 attention will not pass atol 1e-2; compare with a relative/cosine-similarity check or atol ~5e-2, and document the chosen bar in the config comment.

**Inputs to think about.** Prefill-shaped where quant attention shines: batch 1–4, heads 32, head_dim 128, sequence 2k–8k, causal mask. Long sequence is the point — the speedup grows with S.

**Difficulty.** Medium — math is hidden in the kernel; the real work is a fair correctness bar (don't let a loose tolerance hide a wrong layout) and matching the kernel's expected causal/scale conventions.

**Refs.** SageAttention papers (v1/v2, smooth-quantized attention). Hub: https://huggingface.co/kernels-community/sage-attention
