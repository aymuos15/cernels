---
tags:
- kernel
library_name: kernels
license: apache-2.0
---

# Separable Gaussian blur

Separable 2D Gaussian blur, matching `kornia.filters.gaussian_blur2d` (two 1D passes).

Built with [kernel-builder](https://github.com/huggingface/kernel-builder) as a native `torch.library` op (AOT, loadable via the [`kernels`](https://github.com/huggingface/kernels) library). Reference op: kornia `gaussian_blur2d`.

## Benchmark (NVIDIA GB10 / sm_121, aarch64, bf16)

Mean latency per workload — `op_eager` / `op_compile` are the torch reference (eager and `torch.compile`); `custom` is this kernel. The custom kernel runs at **2.085 ms** — **4.86×** vs `op_compile`, verified against the reference.

| workload | mean (ms) |
|---|---|
| `op_eager` (reference) | 5.528 |
| `op_compile` | 10.139 |
| `custom` | 2.085 |

## Usage

```python
import torch
from kernels import get_kernel

kernel = get_kernel("aymuos15/gaussian-blur-2d", version=1)
# call the kernel's registered op, e.g. kernel.<op>(...)
```

> Note: the published build targets **aarch64 / CUDA / sm_121** (NVIDIA GB10). Other platforms need the full build matrix.
