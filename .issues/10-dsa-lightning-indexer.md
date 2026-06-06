# 10 · DeepSeek Sparse Attention — lightning indexer

**Why compile loses.** DSA (DeepSeek V3.2-Exp, Sept 2025) is the canonical "select-then-compute" op. A lightweight **lightning indexer** scores every past token against the current query with a multi-head ReLU-gated dot product, takes **top-k** (k≈2048), and only then runs full attention over that sparse subset — turning per-layer core attention from O(L²) to O(Lk). The top-k selection produces a **data-dependent index set**, which is exactly what `torch.compile` cannot fuse across: it graph-breaks on the value-dependent gather, then falls back to eager for the sparse attention. A fused kernel owns indexer → top-k → gather → attention end to end.

**Source.** Write our own (no clean Hub kernel yet); reference math from DeepSeek-V3.2-Exp and the SGLang implementation. The indexer alone is a great first standalone kernel — small and self-contained.

**Config sketch.** `Config` (non-Hub), dtype bf16. Split into two trials: (a) **indexer only** — `baseline` = dense scores `relu(q_idx @ k_idx.T)` gated and summed over index heads, then `topk`; benchmark a fused indexer+topk kernel as `custom`. (b) **full DSA** — `baseline` = dense SDPA; `custom` = indexer + gather + sparse attention. `verify` compares the selected-index set / attention output at atol ~2e-2; pin the indexer weights so baseline and kernel select the same top-k.

**Inputs to think about.** batch 1–4, heads 64, head_dim 128, sequence 8k–64k (the win grows with L), k=2048. Long context is the entire point.

**Difficulty.** High — the indexer math and the gather/sparse-attention plumbing are both non-trivial. Start with indexer-only (issue 18 covers the generic top-k+gather pattern this specializes).

**Refs.** DeepSeek-V3.2-Exp: https://github.com/deepseek-ai/DeepSeek-V3.2-Exp · SGLang DSA: https://shawnding.medium.com/deepseek-sparse-attention-and-its-implementation-in-sglang-b0bb907c375a · LMSYS day-0: https://www.lmsys.org/blog/2025-09-29-deepseek-V32/
