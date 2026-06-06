# 36 · Gated DeltaNet (linear-attention delta-rule scan)

**Why compile loses.** Qwen3-Next's hybrid replaces most full-attention layers with **Gated DeltaNet** — a linear-attention layer that maintains a matrix-valued recurrent state updated by the *delta rule* (`S ← S(I − β k kᵀ) + β v kᵀ`) with an input-dependent gate, computed as a chunked parallel scan. It's a sequential recurrence over chunks: `torch.compile` cannot turn the data-dependent scan into a fused chunked kernel and falls back to an eager loop that round-trips the state to HBM each chunk. The chunked-scan kernel keeps the state in SRAM across the chunk. This is the SOTA linear-attention op (distinct from the flash-linear-attention we deferred) and the engine of the most efficient 2026 hybrids.

**Source.** Write our own (chunked delta-rule scan, Triton), reference the Gated DeltaNet / Qwen3-Next modeling code.

**Config sketch.** `Config` (non-Hub), dtype bf16 (fp32 state accumulation). `baseline` = a reference chunked or per-step delta-rule recurrence with the gate. `custom` = the fused chunked-scan kernel. Inputs: q/k/v (b, h, s, d), the delta gate `β` (s,), the decay/forget gate, chunk size. `verify` at atol ~2e-2 vs the reference scan.

**Inputs to think about.** b=4, heads 16, head_dim 128, sequence 4k–32k, chunk 64/128. Long sequence is where the linear-attention scan beats quadratic attention.

**Difficulty.** High — the chunked delta-rule math (intra-chunk parallel + inter-chunk recurrent correction) is intricate; validate against a slow per-step reference first. Related to issues 03/04 (SSM/RWKV scans).

**Refs.** Gated DeltaNet; Qwen3-Next hybrid: https://huggingface.co/Qwen/Qwen3-Next-80B-A3B-Instruct · gated attention (NeurIPS 2025): https://towardsdatascience.com/neurips-2025-best-paper-review-qwens-systematic-exploration-of-attention-gating/
