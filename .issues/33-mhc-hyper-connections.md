# 33 · mHC — Manifold-Constrained Hyper-Connections

**Why compile loses.** DeepSeek V4 replaces the plain residual add (`x = x + sublayer(x)`) with **Hyper-Connections**: the residual stream is widened into `n` parallel streams, and each layer learns a small mixing matrix that routes/combines across streams (with a manifold constraint on the weights) instead of a single additive skip. This turns every residual junction into a tiny learned matmul + recombination across the stream dimension. `torch.compile` will keep these as separate small ops; a fused kernel does the per-token stream-mixing in one pass. Small per-op but it runs at *every* layer junction, so the launch-overhead and bandwidth savings add up.

**Source.** Write our own; reference from the DeepSeek-V4 mHC description (Hyper-Connections, ICLR-era work, + V4's manifold constraint).

**Config sketch.** `Config` (non-Hub), dtype bf16. `baseline` = the eager hyper-connection: expand to `n` streams, apply the `n×n` (plus input/output) mixing weights, recombine — the explicit small-matmul version. `custom` = a fused stream-mixing kernel. Inputs: hidden `x` (tokens, n_streams, hidden), the mixing weights, sublayer output. `verify` allclose at atol ~2e-2.

**Inputs to think about.** tokens=8192, hidden=4096, n_streams=2–4. Sweep n_streams; the mixing cost grows with it.

**Difficulty.** Medium — the math is small and fixed; the value is fusing the expand/mix/recombine so the widened residual stream isn't round-tripped to HBM at every layer.

**Refs.** DeepSeek-V4 mHC guide: https://medium.com/mitb-for-all/deepseek-v4-beyond-basics-a-practical-guide-to-mhc-csa-hca-and-muon-bf40c9863ef8 · Raschka on mHC: https://magazine.sebastianraschka.com/p/recent-developments-in-llm-architectures
