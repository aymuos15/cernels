---
tags:
- kernel
library_name: kernels
license: apache-2.0
---

# Qwen3-Next MoE experts

Fused grouped-GEMM MoE for Qwen3-Next: 512 routed experts, top-10, plain SiLU SwiGLU, softmax-top-k router with renorm, and an always-on sigmoid-gated shared expert.

Built with [kernel-builder](https://github.com/huggingface/kernel-builder) as a native `torch.library` op (AOT, loadable via the [`kernels`](https://github.com/huggingface/kernels) library). Reference op: transformers `Qwen3NextSparseMoeBlock`.

## Benchmark (NVIDIA GB10 / sm_121, aarch64, bf16)

Mean latency per workload — `op_eager` / `op_compile` are the torch reference (eager and `torch.compile`); `custom` is this kernel. The custom kernel runs at **27.930 ms** — **2.70×** vs `op_compile`, verified against the reference.

| workload | mean (ms) |
|---|---|
| `op_eager` (reference) | 74.609 |
| `op_compile` | 75.398 |
| `custom` | 27.930 |

## Usage

```python
import torch
from kernels import get_kernel

kernel = get_kernel("aymuos15/qwen3-next-moe-experts", version=1)
# call the kernel's registered op, e.g. kernel.<op>(...)
```

> Note: the published build targets **aarch64 / CUDA / sm_121** (NVIDIA GB10). Other platforms need the full build matrix.
