# 31 · Cross-layer Shared-KV attention

**Why compile loses.** Gemma 4 cuts KV-cache memory two ways: GQA (share KV across heads within a layer) *and* **Shared KV Cache** — multiple layers reuse one layer's KV cache instead of each computing its own. The attention kernel must read K/V produced by a *different* layer, which is a memory-layout/indirection pattern, not a math change. `torch.compile` sees the cross-layer cache read as an opaque gather and can't fuse it into attention; the win is avoiding the redundant K/V projections and the extra cache traffic. Low arithmetic, high memory-layout payoff.

**Source.** Write our own; reference from the Gemma 4 architecture (shared-KV / cross-layer KV sharing, à la YOCO / CLA).

**Config sketch.** `Config` (non-Hub), dtype bf16. `baseline` = standard per-layer attention where each layer projects its own K/V. `custom` = attention that reads a shared K/V cache for the layers configured to share, skipping their K/V projections. Inputs: q per layer, a shared K/V tensor, the layer→cache mapping. `verify` at atol ~2e-2; the point is identical output with fewer projections + less cache, so also report KV-memory saved.

**Inputs to think about.** Model-shaped: 32 layers with a sharing pattern (e.g. groups of 2–4 layers sharing one cache), heads 32, head_dim 128, sequence 8k–32k.

**Difficulty.** Medium — more a cache-layout and scheduling kernel than a math kernel; correctness is easy (same attention, shared inputs). The interesting work is proving the memory/bandwidth win.

**Refs.** Gemma 4 shared KV cache + PLE: https://botmonster.com/ai/gemma-4-architecture-per-layer-embeddings-shared-kv-cache-dual-rope/ · CLA / YOCO cross-layer KV sharing.
