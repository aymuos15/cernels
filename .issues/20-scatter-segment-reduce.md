# 20 · scatter_reduce / segment reduce (ragged)

**Why compile loses.** Ragged reductions — `scatter_reduce`, segment-sum/mean over variable-length groups (GNN message passing, embedding-bag, MoE token combine) — combine two compile pain points: **data-dependent segment boundaries** and **atomic-contended scatters**. Inductor's lowering of scatter-with-atomics is weak, and the variable segment counts resist fusion. A kernel that sorts-by-key once and does a segmented reduction (or uses well-laid-out atomics) beats it.

**Source.** Write our own; `lib` baseline = `torch.scatter_reduce` / `torch_scatter.segment_csr` if available.

**Config sketch.** `Config` (non-Hub), dtype fp16/fp32. `baseline` = `torch.zeros(...).scatter_reduce_(0, index, src, reduce="sum")` (or segment_csr). `custom` = a segmented-reduction kernel. Inputs: `src` (N, D) features, `index` (N,) segment ids, number of segments. `verify` allclose at atol ~1e-2 (watch fp accumulation order for sum).

**Inputs to think about.** N=1M rows, D=128 feature width, segments=10k–100k with **skewed segment sizes** (power-law) — skew is what stresses atomic contention and load balance.

**Difficulty.** Medium — load-balancing skewed segments is the interesting part; a naive atomic scatter is easy but contention-bound, a sort-then-segment approach is the win.

**Refs.** PyTorch troubleshooting (scatter/atomics lowering): https://docs.pytorch.org/docs/stable/torch.compiler_troubleshooting.html
