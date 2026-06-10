---
tags:
- kernel
library_name: kernels
license: apache-2.0
---

# DeepSeek-OCR-2 MoE experts

Fused routed-experts MoE for DeepSeek-OCR-2 (deepseek-community/DeepSeek-OCR-2 text model): 64 routed experts, top-6, plain SiLU SwiGLU, router-weight scale + scatter-add. Routing (fp32 softmax + greedy top-6 scaled by `routed_scaling_factor`, `DeepseekOcr2TextMoe.route_tokens_to_experts`) and the 2 shared experts happen outside the op; both entry points consume `(hidden_states, top_k_index, top_k_weights)`: `deepseek_ocr2_moe_experts` (grouped GEMM, prefill shapes) and `deepseek_ocr2_moe_experts_decode` (fused gather-GEMV for n_tokens ~ 1–4 — two persistent-shape kernels, no per-expert cuBLAS loop, no D2H sync). Constant-swap port of `aymuos15/cohere2-moe-experts`.

Built with [kernel-builder](https://github.com/huggingface/kernel-builder) as native `torch.library` ops (AOT, loadable via the [`kernels`](https://github.com/huggingface/kernels) library). Reference op: transformers `DeepseekOcr2TextExperts`.

## Benchmark (NVIDIA GB10 / sm_121, aarch64, bf16)

Mean latency per workload — `op_eager` / `op_compile` are the torch reference (eager and `torch.compile`); `custom` is this kernel. Hidden 1280, per-expert intermediate 896, 64 experts top-6.

Prefill shape — n_tokens = 2048, `deepseek_ocr2_moe_experts`: **4.261 ms**, **2.26×** vs `op_compile`, verified.

| workload | mean (ms) |
|---|---|
| `op_eager` (reference) | 9.253 |
| `op_compile` | 9.610 |
| `custom` | 4.261 |

Decode shape — n_tokens = 1, `deepseek_ocr2_moe_experts_decode`: **0.191 ms**, **4.35×** vs `op_compile`, verified.

| workload | mean (ms) |
|---|---|
| `op_eager` (reference) | 0.784 |
| `op_compile` | 0.830 |
| `custom` | 0.191 |

## Usage

```python
import torch
from kernels import get_kernel

kernel = get_kernel("aymuos15/deepseek-ocr2-moe-experts", version=1)
# call the kernel's registered op, e.g. kernel.<op>(...)
```

> Note: the published build targets **aarch64 / CUDA / sm_121** (NVIDIA GB10). Other platforms need the full build matrix.
