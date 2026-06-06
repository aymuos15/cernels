# 19 · unique / nonzero / mask compaction

**Why compile loses.** These are the textbook unbacked-SymInt ops: `torch.unique`, `torch.unique_consecutive`, `nonzero`, and boolean-mask compaction (`x[x > 0]`) all produce an **output whose length is the number of elements that satisfy a runtime condition**. `torch.compile` cannot determine the size at trace time → graph break (or hard error under `fullgraph=True`). Common in tokenizer/dedup/sparse-prep pipelines. A stream-compaction kernel (prefix-sum + scatter) does it in one pass.

**Source.** Write our own (prefix-sum + compaction); `lib` baseline = the corresponding `torch` op (`torch.unique` / `torch.nonzero`).

**Config sketch.** `Config` (non-Hub), dtype int32/int64 or bool mask. Pick one concrete op per trial. `baseline` = `torch.nonzero(x)` (or `torch.unique(x)`). `custom` = a CUB-style stream-compaction kernel. Inputs: a tensor with a tunable density of satisfying elements. **Override `verify`** — variable-length output; compare sorted result sets, not `allclose`.

**Inputs to think about.** N=1M–16M elements, density sweeps (1%, 10%, 50% nonzero). Density matters: the win and the work both scale with how many survive.

**Difficulty.** Low-medium — stream compaction is standard (Thrust/CUB territory); beating `torch`'s own CUB call is the real bar, so focus on fusing the predicate so you don't make two passes.

**Refs.** PyTorch dynamic-shapes docs on unbacked SymInts: https://docs.pytorch.org/docs/main/user_guide/torch_compiler/torch.compiler_dynamic_shapes.html
