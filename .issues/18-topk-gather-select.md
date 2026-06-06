# 18 · Top-k + gather (select-then-compute)

**Why compile loses.** The general pattern behind issues 10/11/12 and MoE routing: score → **top-k** → gather the selected rows → dense compute. The gather index set is data-dependent, so `torch.compile` graph-breaks at the gather and can't fuse the selection into the downstream matmul. Worth a standalone issue because it's the reusable primitive — land this kernel once and the sparse-attention and MoE issues all build on it.

**Source.** Write our own (fused top-k + gather, Triton or CUDA).

**Config sketch.** `Config` (non-Hub), dtype bf16. `baseline` = `torch.topk(scores, k)` then `torch.gather`/index then a matmul over the gathered rows. `custom` = a kernel fusing top-k selection + gather (+ optionally the following GEMM). Inputs: `scores` (rows, candidates), value matrix to gather from, `k`. `verify` at atol ~2e-2; pin scores so baseline and kernel select the same indices (ties make top-k non-deterministic).

**Inputs to think about.** rows=4096, candidates=8k–64k, k=256/2048. Large candidate pool with small k is where the fused gather beats the materialize-then-index path.

**Difficulty.** Medium — top-k itself is the tricky kernel (bitonic / radix-select); the gather fusion is the value-add over calling `torch.topk`.

**Refs.** Specializes into issue 10 (DSA indexer), issue 11 (NSA block select), issue 06 (MoE routing).
