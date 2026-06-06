# 07 · FP8 quantized GEMM

**Why compile loses.** This is the clearest structural win in the backlog. `torch.compile` can fuse the dequantization pointwise ops *around* a quantized matmul but it cannot fuse *into* the core quantized GEMM — that kernel is opaque to it. A fine-grained FP8 matmul (per-tensor or per-block scales, FP8 E4M3 inputs accumulating in fp32, on Hopper/Ada tensor cores) runs at roughly 2× the fp16 throughput with a fraction of the weight bandwidth. Compile has no path to this.

**Source.** `kernels-community/finegrained-fp8` (alternative: `kernels-community/fp8-fbgemm`).

**Config sketch.** `HubConfig`, dtype fp16/bf16 for the activation side. `baseline` is a plain `torch.matmul` of `x` (fp16) against a dequantized weight (`w_fp8.to(fp16) * scale`), i.e. the math the quant kernel approximates. `op` is the kernel's FP8 GEMM entry. Not `out_arg`. Inputs: activation `x` (M, K) in fp16, weight `w` pre-quantized to FP8 (N, K) plus its scale(s). `verify` must be loose — FP8 will not pass atol 1e-2; use a relative-error / cosine bar (e.g. rel-err < 2%) and document it. Requires an FP8-capable GPU (cc ≥ 8.9 / Hopper); guard the config so it skips cleanly on older arch.

**Inputs to think about.** LLM-FFN-shaped: M=4096, K=4096, N=11008. Large K/N is where the tensor-core FP8 path dominates.

**Difficulty.** Medium-high — quantizing the weight to match the kernel's expected scale granularity (per-tensor vs per-block) and setting an honest correctness bar are the work. Confirm the GPU supports FP8 before benchmarking.

**Refs.** DeepSeek fine-grained FP8; Transformer Engine FP8 recipes. Hub: https://huggingface.co/kernels-community/finegrained-fp8
