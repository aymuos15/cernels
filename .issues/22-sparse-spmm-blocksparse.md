# 22 · Sparse SpMM / block-sparse matmul

**Why compile loses.** Sparse layouts (CSR/COO SpMM, block-sparse GEMM) are outside Inductor's world — it reasons about dense tensors and can't lower or fuse sparse formats, so it falls back entirely. Block-sparse matmul (fixed-size nonzero blocks, e.g. for pruned/structured-sparse models and block-sparse attention) is the practical target: regular enough for tensor cores, sparse enough to skip most of the FLOPs. compile offers nothing here.

**Source.** Write our own (block-sparse GEMM, Triton's blocksparse as a reference) or bind an existing block-sparse lib as `lib`.

**Config sketch.** `Config` (non-Hub), dtype bf16. `baseline` = dense `torch.matmul` of the equivalent dense weight (the FLOPs you're skipping), or a masked dense matmul. `custom` = block-sparse GEMM driven by a block mask. Inputs: dense activation `x` (M, K), block-sparse weight given as nonzero blocks + a block index/mask, block size (e.g. 64 or 128), sparsity ratio. `verify` allclose at atol ~2e-2 against the dense-equivalent product.

**Inputs to think about.** M=4096, K=4096, N=11008, block 128, sparsity 50–90%. Higher sparsity = bigger win but the block structure must stay tensor-core-friendly.

**Difficulty.** Medium-high — the block-sparse tiling and index handling are the work; correctness is easy (compare to the dense-equivalent).

**Refs.** Triton block-sparse matmul; structured-sparsity (2:4) as a hardware-supported special case.
