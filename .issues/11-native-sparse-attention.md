# 11 · Native Sparse Attention (NSA)

**Why compile loses.** NSA (Feb 2025, hardware-aligned) is block-wise sparse attention with three branches — **compressed** (coarse token summaries), **selected** (top blocks retained), and **sliding window** (local) — merged by learned gate signals. The selected branch is a data-dependent block gather (same compile-hostile pattern as DSA, but block-granular instead of token-granular), and the three-branch merge is a fusion compile won't synthesize. So both the sparsity and the gating defeat it.

**Source.** Write our own; reference from the NSA paper (and the public `native-sparse-attention` Triton reference if used as a cross-check).

**Config sketch.** `Config` (non-Hub), dtype bf16. `baseline` = dense SDPA (GQA). `custom` = the three-branch NSA kernel. Inputs: q/k/v in GQA layout, block size (e.g. 64), compression ratio, number of selected blocks, sliding-window size, and the per-branch gate weights. `verify` at atol ~2e-2 with pinned gates and pinned block selection so baseline and kernel agree on which blocks are kept.

**Inputs to think about.** batch 1–4, GQA (e.g. 64 q-heads / 4 kv-heads), head_dim 128, sequence 8k–64k, block 64, 16 selected blocks.

**Difficulty.** High — three branches plus gating is the most complex attention variant in the backlog. Implement branches independently, validate each against a dense mask, then fuse.

**Refs.** NSA paper: https://arxiv.org/pdf/2502.11089 · bycloud overview: https://mail.bycloud.ai/p/deepseek-s-native-sparse-attention
