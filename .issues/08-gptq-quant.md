# 08 · GPTQ 4-bit matmul

**Why compile loses.** GPTQ packs weights to 4-bit with group-wise scales/zeros. Inference fuses dequant-and-matmul: unpack 4-bit weights, apply per-group scale/zero, and multiply, all in one kernel so the 4-bit weights stay compressed until the last moment. `torch.compile` cannot enter that fused quant GEMM; a baseline that dequantizes to fp16 first pays full weight bandwidth. The kernel's win is mostly memory (4× smaller weights) and is decode/low-batch-bound, which makes it a different shape of win from the FP8 issue (compute-bound) — worth having both.

**Source.** `kernels-community/quantization-gptq` (related: `kernels-community/quantization-eetq`).

**Config sketch.** `HubConfig`, dtype fp16. `baseline` dequantizes the packed weight to fp16 (unpack 4-bit → `(w - zero) * scale` per group) and does `x @ w_deq.T`. `op` is the kernel's GPTQ matmul entry. Not `out_arg`. Inputs: activation `x` (M, K) fp16, packed `qweight` (int32-packed), `qzeros`, `scales`, `g_idx`/group-size metadata, bits=4. `verify` at atol ~2e-2 against the dequant baseline (same scales/zeros feed both).

**Inputs to think about.** Decode regime: M=1 or 8 (small batch), K=4096, N=11008, group-size 128. Low M is the point — this win is bandwidth-bound, so keep batch small.

**Difficulty.** Medium — the packing/group-metadata layout must exactly match what the kernel expects; reuse an existing GPTQ-quantized weight or quantize one with the kernel's own packer rather than hand-rolling.

**Refs.** GPTQ paper (Frantar et al.); AutoGPTQ packing. Hub: https://huggingface.co/kernels-community/quantization-gptq
