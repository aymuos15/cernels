# 13 · MXFP4 / NVFP4 microscaling GEMM

**Why compile loses.** Block-scaled 4-bit floating point (OCP MX format: one shared exponent per block of 32 values). `torch.compile` cannot enter a quantized matmul at all — it's a hardware-native tensor-core path. QuTLASS (on CUTLASS) reports **3.6× layer-wise speedup on B200 and 6× on RTX5090**, ~2–4× end-to-end. Biggest single inference win in the whole backlog, but **hardware-gated**: needs Blackwell (B200 / RTX50xx) or AMD MI355 with native FP4; older arch can't run it.

**Source.** QuTLASS (CUTLASS-based 4-bit kernel lib); MR-GPTQ (ICLR 2026) for calibration. Either bind QuTLASS or write a CUTLASS FP4 GEMM.

**Config sketch.** `HubConfig`/`Config`, fp16/bf16 activations. `baseline` = `torch.matmul` of `x` against a dequantized weight (`w_mxfp4 → fp16 × block_scale`). `op`/`custom` = the FP4 GEMM. Inputs: `x` (M, K) fp16, weight pre-quantized to MXFP4 with per-32 block scales (N, K). **Guard the config to skip on non-FP4 hardware** (check `torch.cuda.get_device_capability()`; needs cc ≥ 10.0 Blackwell, or NVFP4 path on cc 12.0). `verify` with a relative/cosine bar (rel-err < ~3%), documented — FP4 won't pass atol 1e-2.

**Inputs to think about.** LLM-FFN shapes: M=4096, K=4096, N=11008. Compare MXFP4 (block 32, e8m0 scale) vs NVFP4 (block 16, fp8 scale) if both available.

**Difficulty.** Medium-high once on the right GPU; otherwise blocked. The work is matching the kernel's block-scale layout and an honest correctness bar.

**Refs.** QuTLASS: https://medium.com/@bnjmn_marie/qutlass-efficient-inference-with-4-bit-models-for-blackwell-gpus-970caad4a153 · MXFP4 spec/overview: https://www.spheron.network/blog/mxfp4-microscaling-quantization-gpu-cloud/ · NVFP4 vs MXFP4: https://insiderllm.com/guides/fp4-inference-llamacpp-nvfp4-mxfp4/
