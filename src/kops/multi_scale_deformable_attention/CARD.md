---
tags:
- kernel
library_name: kernels
license: apache-2.0
---

# Multi-scale deformable attention

Multi-scale deformable attention (the `multi_scale_deformable_attention` op from Deformable DETR) — bilinear sampling over multi-level feature maps with learned offsets and attention weights.

Built with [kernel-builder](https://github.com/huggingface/kernel-builder) as a native `torch.library` op (AOT, loadable via the [`kernels`](https://github.com/huggingface/kernels) library). Reference op: transformers `multi_scale_deformable_attention` (Deformable DETR).

## Benchmark (NVIDIA GB10 / sm_121, aarch64, bf16)

Mean latency per workload — `op_eager` / `op_compile` are the torch reference (eager and `torch.compile`); `custom` is this kernel. The custom kernel runs at **0.026 ms** — **34.90×** vs `op_compile`, verified against the reference.

| workload | mean (ms) |
|---|---|
| `op_eager` (reference) | 0.637 |
| `op_compile` | 0.904 |
| `hub` | 0.036 |
| `custom` | 0.026 |

## Usage

```python
import torch
from kernels import get_kernel

kernel = get_kernel("aymuos15/multi-scale-deformable-attention", version=1)
# call the kernel's registered op, e.g. kernel.<op>(...)
```

> Note: the published build targets **aarch64 / CUDA / sm_121** (NVIDIA GB10). Other platforms need the full build matrix.
