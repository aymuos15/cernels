# 09 · Custom fused RMSNorm (write our own)

**Why this one.** The custom-kernel track, modeled on the existing RoPE win. RMSNorm computes a row-wise mean-square, a reciprocal-sqrt, then scales by a learned weight. A naive path makes several HBM passes (reduce, then normalize, then scale); a fused kernel does it in one pass with the reduction in shared memory. HF's own agent demo measured ~1.88–1.94x over their reference on this exact kernel, so it's a proven, achievable target — and unlike the Hub issues, we write and own the CUDA, the same way `src/kops/rope.cu` is ours.

**Source.** Write our own under `src/kops/` — a `rmsnorm.cu` + `rmsnorm.py` pair compiled with `load_inline`, exactly like `src/kops/rope.py`/`rope.cu`. Wire it as the `custom` workload, with `lib` pointed at `kernels-community/rmsnorm` so we benchmark eager vs compile vs Hub vs ours in one run.

**Config sketch.** `HubConfig`, dtype fp16. `baseline` is the canonical RMSNorm: `x * rsqrt(mean(x^2, dim=-1, keepdim=True) + eps) * weight` (the transformers `LlamaRMSNorm` math, fp32 reduction). `lib` = `kernels-community/rmsnorm` op. `custom` = our `kernel(x, weight, eps)` from `src/kops/rmsnorm.py`. Inputs: `x` (tokens, hidden) and `weight` (hidden,). Standard `verify` allclose at atol ~1e-2; do the reduction in fp32 inside the kernel to pass it.

**Inputs to think about.** tokens=4096 (batch×seq), hidden=4096 — a transformer-block-shaped row reduction. Also try hidden=8192 to stress the shared-memory reduction.

**Kernel notes.** One block per row (or a warp-per-row for small hidden); accumulate sum-of-squares in fp32 via a warp/block reduction; vectorize loads (float4) for bandwidth; apply `weight` in the same pass. This is genuinely fusion+reduction, not pure pointwise, which is why it can edge out compile.

**Difficulty.** Medium — the CUDA reduction is standard but easy to get wrong on tail elements when hidden isn't a multiple of the vector width; mirror the structure of `src/kops/rope.cu` for the build/binding glue.

**Refs.** RMSNorm paper (Zhang & Sennrich); HF "Custom Kernels for All from Codex and Claude" (the ~1.9x RMSNorm result): https://huggingface.co/blog/custom-cuda-kernels-agent-skills
