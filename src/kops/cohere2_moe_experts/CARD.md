---
tags:
- kernel
library_name: kernels
license: apache-2.0
---

# Cohere2Moe experts

Fused MoE experts for Cohere2Moe (CohereLabs/North-Mini-Code-1.0): 128 routed experts, top-8, plain SiLU SwiGLU, router-weight scale + scatter-add, no shared expert. Routing (sigmoid top-k, `Cohere2MoeTopKRouter`) happens outside the op; both entry points consume `(hidden_states, top_k_index, top_k_weights)`: `cohere2_moe_experts` (grouped GEMM, prefill shapes) and `cohere2_moe_experts_decode` (fused gather-GEMV for n_tokens ~ 1–4 — two persistent-shape kernels, no per-expert cuBLAS loop, no D2H sync).

Built with [kernel-builder](https://github.com/huggingface/kernel-builder) as native `torch.library` ops (AOT, loadable via the [`kernels`](https://github.com/huggingface/kernels) library). Reference op: transformers `Cohere2MoeExperts`.

## Benchmark (NVIDIA GB10 / sm_121, aarch64, bf16)

Mean latency per workload — `op_eager` / `op_compile` are the torch reference (eager and `torch.compile`); `custom` is this kernel. Hidden 2048, per-expert intermediate 768, 128 experts top-8.

Prefill shape — n_tokens = 2048, `cohere2_moe_experts`: **9.990 ms**, **2.11×** vs `op_compile`, verified.

| workload | mean (ms) |
|---|---|
| `op_eager` (reference) | 21.269 |
| `op_compile` | 21.085 |
| `custom` | 9.990 |

Decode shape — n_tokens = 1, `cohere2_moe_experts_decode`: **0.341 ms**, **3.58×** vs `op_compile`, verified. At n_tokens = 4 (ad-hoc probe, same timer): custom 1.228 ms vs op_compile 4.309 ms (**3.51×**).

| workload | mean (ms) |
|---|---|
| `op_eager` (reference) | 1.194 |
| `op_compile` | 1.222 |
| `custom` | 0.341 |

## Usage

```python
import torch
from kernels import get_kernel

kernel = get_kernel("aymuos15/cohere2-moe-experts", version=1)
# call the kernel's registered op, e.g. kernel.<op>(...)
```

> Note: the published build targets **aarch64 / CUDA / sm_121** (NVIDIA GB10). Other platforms need the full build matrix.
