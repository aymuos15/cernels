# 23 · Ragged / variable-length batched matmul (grouped GEMM)

**Why compile loses.** A batch of matmuls with **different per-item sizes** (variable sequence lengths, per-expert token counts). The data-dependent loop count forces Dynamo to unroll extensively — blowing up compile time and producing little fused — or to pad to a fixed max, wasting FLOPs on padding. A grouped/varlen GEMM processes all items in one launch over a ragged layout with no padding. This is the matmul cousin of issue 06 (MoE) and the engine behind varlen FlashAttention.

**Source.** Write our own (grouped GEMM over a ragged/cu_seqlens layout), or bind CUTLASS grouped GEMM as `lib`.

**Config sketch.** `Config` (non-Hub), dtype bf16. `baseline` = padded dense batched matmul (pad to max length, `torch.bmm`, then mask) — the wasteful path compile is stuck with. `custom` = grouped GEMM driven by `cu_seqlens` offsets. Inputs: concatenated A/B in a ragged layout, `cu_seqlens` (group offsets), shared inner dims. `verify` allclose at atol ~2e-2 against the padded result (with padding removed).

**Inputs to think about.** groups=64, lengths drawn from a skewed distribution (e.g. 128–4096), inner dim K=128. Skew is the point — padding to max wastes most of the work.

**Difficulty.** Medium-high — the ragged offset handling and load balancing across uneven groups. Related: issue 06 (MoE grouped GEMM) is the same machinery with a routing front-end.

**Refs.** CUTLASS grouped GEMM; varlen FlashAttention `cu_seqlens` layout. PyTorch unroll/compile-time issue with data-dependent loops: https://docs.pytorch.org/docs/stable/torch.compiler_troubleshooting.html
