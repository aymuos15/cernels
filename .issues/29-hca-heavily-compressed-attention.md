# 29 · HCA — Heavily Compressed Attention (DeepSeek V4)

**Why compile loses.** The long-range companion to CSA (issue 28). For the truly distant context, HCA uses a far more aggressive compressor that bundles up to **128 tokens into a single KV entry** (`m′ ≫ m`), then runs *dense* attention over those few compressed representations. The compressor is a learned pooling/projection that `torch.compile` won't fuse into the attention, and the two-tier CSA(local)+HCA(long-range) hybrid is the design that lets V4-Pro run 1M context at ~27% of V3.2's per-token FLOPs and ~10% of its KV cache. A fused compress-then-dense-attend kernel captures it; eager round-trips the compressed cache through HBM.

**Source.** Write our own; reference from DeepSeek-V4 materials. Naturally implemented alongside issue 28 (shared compressor machinery, different `m′` and dense vs sparse second stage).

**Config sketch.** `Config` (non-Hub), dtype bf16. `baseline` = strided learned pooling of distant KV into `s/m′` entries, then dense SDPA over them. `custom` = fused heavy-compressor + dense attention. Inputs: q/k/v, compressor weights, `m′` (e.g. 128). `verify` at atol ~2e-2 with pinned compressor weights.

**Inputs to think about.** b=1, heads 64, head_dim 128, sequence 256k–1M, m′=128 → only a few thousand compressed entries even at 1M. Measure KV memory, not just latency — the cache shrink is the headline.

**Difficulty.** Medium-high — simpler second stage than CSA (dense, not top-k) but the heavy compressor and the CSA/HCA routing (which tokens go to which tier) are the work. Do issue 28 first.

**Refs.** DeepSeek-V4 KV cache: https://knightli.com/en/2026/05/18/deepseek-v4-kv-cache-compressed-attention/ · CSA/HCA explainer: https://www.runyard.dev/blog/deepseek-v4-attention-architecture-explained
