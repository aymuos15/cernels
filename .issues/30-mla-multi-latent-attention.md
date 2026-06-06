# 30 · MLA — Multi-head Latent Attention

**Why compile loses.** DeepSeek's KV-cache compression that compresses along the *head* dimension: K and V are projected down to a shared low-rank latent, cached compressed, and projected back up (absorbed into the query/output projections) at attention time — cutting KV-cache memory bandwidth 5–8× vs GQA. DeepSeek V4 keeps an MLA-style local branch; V3.2/V3 use it throughout. The up-projection-absorbed attention is a specific matmul fusion (`q · (W_up^T K_latent)`) that `torch.compile` won't synthesize, and the cached-latent layout is bespoke. A `flash-mla` Hub kernel exists, so this is a benchmark-the-Hub-kernel issue plus an option to write our own.

**Source.** `kernels-community/flash-mla` (Hub) as `lib`; optionally our own as `custom`.

**Config sketch.** `HubConfig`, dtype bf16. `baseline` = a reference MLA: down-project K/V to latent, cache, up-project, then SDPA (the math the kernel fuses). `op` = flash-mla entry. Inputs: q, compressed KV latent cache, the down/up projection weights, RoPE parts (MLA splits rotary vs non-rotary dims). `verify` at atol ~2e-2.

**Inputs to think about.** b=1–4, heads 128, head_dim 128, latent dim ~512, sequence 8k–64k (decode-shaped, where KV bandwidth dominates).

**Difficulty.** Medium-high — the rotary/non-rotary dimension split and the up-projection absorption are the fiddly parts of the baseline. The Hub kernel does the heavy lifting; correctness-matching the reference is the work.

**Refs.** DeepSeek-V2/V3 MLA; `kernels-community/flash-mla`. V4 hybrid (MLA local + CSA/HCA long-range): see issues 28/29.
