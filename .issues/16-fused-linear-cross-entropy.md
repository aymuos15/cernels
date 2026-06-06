# 16 · Fused linear cross-entropy

**Why compile loses.** The final LM head computes logits = `hidden @ Wᵀ` over a huge vocab, then softmax+cross-entropy. Materializing the full (tokens × vocab) logits to HBM is the memory bottleneck of training. Liger's fused linear-CE computes the projection, softmax, loss, and the input gradient **in chunks without ever materializing the full logits**. `torch.compile` won't synthesize this fusion — it'll keep the giant logits tensor — so the kernel wins big on peak memory (and bandwidth). Runs on any GPU; pure win at large vocab.

**Source.** Write our own (Triton, mirroring Liger's `fused_linear_cross_entropy`), or bind Liger's op as `lib`.

**Config sketch.** `Config` (non-Hub), dtype bf16. `baseline` = `F.cross_entropy(hidden @ W.T, labels)` (the materialize-then-reduce path). `custom` = the fused chunked kernel returning loss (and grad). Inputs: `hidden` (tokens, hidden), `W` (vocab, hidden), `labels` (tokens,). `verify` on the scalar loss at atol ~1e-2; also worth asserting peak memory drop, not just speed (the headline win is memory).

**Inputs to think about.** tokens=8192, hidden=4096, **vocab=128k–256k** (modern tokenizers). Large vocab is the whole point — the logits tensor is what you're avoiding.

**Difficulty.** Medium — Triton chunked reduction over the vocab dimension; the gradient path is the fiddly part. Report memory alongside latency.

**Refs.** Liger fused linear CE: https://github.com/linkedin/Liger-Kernel/blob/main/src/liger_kernel/ops/fused_linear_cross_entropy.py · Liger paper: https://arxiv.org/pdf/2410.10989
