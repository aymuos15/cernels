# 32 · Dual RoPE

**Why compile loses.** Gemma 4 applies **two rotary embeddings** — different frequency bases for the local sliding-window layers vs the global layers (and/or interleaved frequency bands), so position is encoded differently depending on the layer's attention range. It's a direct extension of the RoPE kernel we already beat compile on (1.21x): two rotation passes with distinct cos/sin tables, ideally fused into one pass over q/k. Same reasoning as the original rotary win — multiple small elementwise rotations that compile launches separately and our fused kernel does in one read.

**Source.** Write our own, extending `src/kops/rope.cu`.

**Config sketch.** `Config` (non-Hub), dtype fp16/bf16. `baseline` = apply local-RoPE and global-RoPE (the eager two-pass version matching Gemma 4). `custom` = a fused kernel selecting/blending the two frequency sets in one pass. Inputs: q/k (b, h, s, d), two cos/sin pairs (local base, global base), a per-layer or per-dim selector. `verify` allclose at atol ~1e-2.

**Inputs to think about.** b=2, heads 32, head_dim 128, sequence 4k–16k. Benchmark against the existing single-RoPE config to show the fused dual pass holds the win.

**Difficulty.** Low-medium — the rotary kernel already exists; this adds a second frequency table and the selection logic. Good quick follow-on to the current RoPE work.

**Refs.** Gemma 4 dual RoPE: https://cloudinsight.cc/en/blog/gemma-4-architecture · see existing rotary config + `src/kops/rope.cu`.
