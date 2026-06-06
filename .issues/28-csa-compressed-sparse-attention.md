# 28 · CSA — Compressed Sparse Attention (DeepSeek V4)

**Why compile loses.** The headline op of DeepSeek V4 (Apr 2026) and almost certainly the most impactful, least-kernelized target in this whole backlog. CSA does two compile-hostile things in sequence: (1) a **learned token-level compressor** folds every `m` tokens (m=4) of the KV cache into one entry — a 4× sequence-length reduction; (2) **DSA-style top-k sparse attention** where each query attends only to the top-k *compressed* KV entries. Step 2's top-k is a data-dependent select (#1 graph break); step 1 is a small fused projection compile won't merge into the attention. Together they cut inference FLOPs ~73% and KV cache to ~2% at 1M context. No open kernel exists yet — maximum novelty, and every long-context model will adopt this pattern.

**Source.** Write our own; reference from the DeepSeek-V4 technical materials and the `deepseek_v4` transformers modeling code.

**Config sketch.** `Config` (non-Hub), dtype bf16. Stage it: (a) **compressor** — `baseline` = strided learned pooling of KV `(b,h,s,d) -> (b,h,s/m,d)`; benchmark a fused compressor kernel. (b) **full CSA** — `baseline` = dense SDPA over uncompressed KV; `custom` = compressor + per-query top-k over compressed entries + sparse attention. Inputs: q/k/v, compressor weights, `m=4`, `k` (selected compressed entries). `verify` at atol ~2e-2 with pinned compressor weights so baseline and kernel select the same top-k.

**Inputs to think about.** b=1, heads 64, head_dim 128, sequence 64k–1M (the regime CSA targets), m=4, top-k≈512. The win is enormous at long context; benchmark KV-cache memory alongside latency.

**Difficulty.** High — compressor + data-dependent gather + sparse attention is the most involved attention kernel here. Build on issue 18 (top-k+gather) and issue 10 (DSA indexer); start with the compressor alone.

**Refs.** DeepSeek-V4 CSA/HCA: https://www.marktechpost.com/2026/04/24/deepseek-ai-releases-deepseek-v4-compressed-sparse-attention-and-heavily-compressed-attention-enable-one-million-token-contexts/ · attention deep dive: https://www.intoai.pub/p/what-makes-deekseek-v4-so-good · transformers: https://huggingface.co/docs/transformers/model_doc/deepseek_v4
