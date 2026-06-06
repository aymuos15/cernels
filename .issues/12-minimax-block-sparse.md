# 12 · MiniMax M3 block-sparse attention

**Why compile loses.** MiniMax M3's attention is a single-branch block-selection scheme (simpler than NSA's three branches) that claims **15.6× decode speedup at 1M context vs NSA**. Like the others here it's a data-dependent block gather feeding dense attention — compile graph-breaks on the selection and can't fuse the gather into the attention. The interesting angle vs issue 11 is the simplicity/efficiency trade: one branch, GQA, block selection, which makes it a cleaner first sparse-attention kernel to actually land.

**Source.** Write our own; reference from the MiniMax M3 writeup / model card.

**Config sketch.** `Config` (non-Hub), dtype bf16. `baseline` = dense SDPA (GQA, causal). `custom` = block-select + sparse attention. Inputs: q/k/v (GQA), block size, number of selected blocks per query, selection scores. `verify` at atol ~2e-2 with pinned selection.

**Inputs to think about.** batch 1–2, GQA, head_dim 128, sequence up to 128k–1M (the regime where the 15.6× shows), block 128. Decode-shaped (single query step) is where it's measured.

**Difficulty.** Medium-high — simplest of the three sparse-attention issues (10/11/12); good target once the generic top-k+gather kernel (issue 18) works.

**Refs.** MiniMax M3 attention: https://www.atlascloud.ai/blog/guides/minimax-goes-sparse
