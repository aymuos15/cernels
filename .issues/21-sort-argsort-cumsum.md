# 21 · sort / argsort / cumsum on long rows

**Why compile loses.** These are scan/sort primitives where Inductor has little to add — it leans on cub/thrust under the hood and can't fuse much around them, so there's headroom for a kernel tuned to the specific shape (e.g. row-wise sort for top-k routing, segmented cumsum for prefix operations, argsort for ranking). Not a graph-break case like the others here; rather a "compile gives ~0 benefit, a shape-specialized kernel can win" case.

**Source.** Write our own (shape-specialized scan/sort); `lib` baseline = `torch.sort` / `torch.cumsum`.

**Config sketch.** `Config` (non-Hub), dtype fp16/int32. Pick one per trial. `baseline` = `torch.cumsum(x, dim=-1)` (or `torch.argsort`). `custom` = a block-wise scan / segmented sort kernel. Inputs: `x` (rows, cols). `verify` allclose at atol ~1e-2 for cumsum; for sort/argsort compare the permutation (stable-sort caveats — match torch's tie behavior).

**Inputs to think about.** rows=8192, cols=4096–32768 (long rows, the regime where a register-blocked scan beats a generic library call). Segmented variants (many short rows) are a separate sweep.

**Difficulty.** Medium — beating cub/thrust is a high bar; only pursue where the shape is fixed and special (e.g. always row-wise, known width) so you can out-specialize the general library.

**Refs.** PyTorch troubleshooting (ops with limited fusion): https://docs.pytorch.org/docs/stable/torch.compiler_troubleshooting.html
