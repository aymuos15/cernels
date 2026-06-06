# 15 · Muon optimizer — Newton-Schulz orthogonalization

**Why compile loses.** Muon (the AdamW challenger being adopted in frontier training) orthogonalizes the momentum matrix via **NewtonSchulz5** — 5 iterations of `X = aX + bX(XᵀX) + c(X XᵀX)(XᵀX)`, a tight sequence of matmuls in bf16, fused with the weight/momentum update. The default fused PyTorch Muon eats a **~2.1% slowdown** from this step. It's an optimizer-step op (not a forward op), a regime `torch.compile` barely touches, and the chained matmuls + update are a clean fusion target. Compute-bound, runs on any GPU.

**Source.** Write our own; reference from Keller Jordan's Muon and the Liger Muon kernels (which fuse the weight/momentum update and the Newton-Schulz step).

**Config sketch.** `Config` (non-Hub), dtype bf16. `baseline` = reference PyTorch Muon step: momentum update, then 5 Newton-Schulz iterations (each two matmuls), then the orthogonalized update with the spectral-norm scaling. `custom` = a fused kernel for the Newton-Schulz iterations (and optionally the update). Inputs: a momentum/gradient matrix `G` (m, n) and the Newton-Schulz coefficients. `verify` allclose on the orthogonalized output at atol ~2e-2 (bf16).

**Inputs to think about.** Hidden-layer weight shapes: (4096, 4096), (8192, 2048), (2048, 8192) — Muon targets 2D hidden weights. Try square and tall/wide to stress the XᵀX shape.

**Difficulty.** Medium — the math is fixed and small; the win is fusing the 5 iterations to keep `X`/`XᵀX` in registers/SRAM instead of round-tripping HBM each step.

**Refs.** Muon (Keller Jordan): https://kellerjordan.github.io/posts/muon/ · NorMuon: https://arxiv.org/pdf/2510.05491
