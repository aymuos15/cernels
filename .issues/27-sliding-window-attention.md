# 27 · Sliding-window / banded attention

**Why compile loses.** Gemma 4 runs local **sliding-window attention** (512 tokens on E-series, 1024 on 26B/31B) in 5 of every 6 layers; gpt-oss alternates a 128-token window with full attention. The window is a banded mask over the QK^T scores — flash-style attention that only computes the diagonal band and skips the rest. `torch.compile` can't tile attention at all, and a dense+mask baseline wastes O(S²) work computing scores it then throws away. A banded flash kernel computes only the in-window tiles, so the win grows with sequence length. Since these are the *majority* of layers in the leading 2026 models, this is one of the highest-leverage kernels in the backlog.

**Source.** Write our own (flash-attention with a window-bounded tile loop).

**Config sketch.** `Config` (non-Hub), dtype bf16. `baseline` = `scaled_dot_product_attention` with a sliding-window mask (or `F.sdpa` with `is_causal` + band). `custom` = banded flash kernel that only iterates KV tiles within `[i-w, i]`. Inputs: q/k/v (b, h, s, d), window size `w`, causal flag. `verify` allclose at atol ~2e-2 vs the masked dense reference.

**Inputs to think about.** b=2, heads 32, head_dim 128, window 512/1024, sequence 8k–32k. Large S with small w is exactly where skipping out-of-band tiles dominates.

**Difficulty.** Medium — a flash kernel with a restricted KV-tile range; simpler than full sparse attention. Natural base for issue 26 (add QK-norm/RoPE prologue) and issue 28/29 (compressed variants).

**Refs.** Gemma 4 hybrid local/global 5:1; gpt-oss banded attention. Mistral sliding-window attention is the original.
