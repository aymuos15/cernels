---
tags:
- kernel
library_name: kernels
license: apache-2.0
---

# RMSNorm

Root-mean-square layer norm, matching `torch.nn.functional.rms_norm` (fp32 reduction).

Built with [kernel-builder](https://github.com/huggingface/kernel-builder) as a native `torch.library` op (AOT, loadable via the [`kernels`](https://github.com/huggingface/kernels) library). Reference op: `torch.nn.functional.rms_norm`.

## Benchmark (NVIDIA GB10 / sm_121, aarch64, bf16)

Mean latency per workload — `op_eager` / `op_compile` are the torch reference (eager and `torch.compile`); `custom` is this kernel. The custom kernel runs at **0.128 ms** — **1.23×** vs `op_compile`, verified against the reference.

| workload | mean (ms) |
|---|---|
| `op_eager` (reference) | 0.147 |
| `op_compile` | 0.157 |
| `custom` | 0.128 |

## Usage

```python
import torch
from kernels import get_kernel

kernel = get_kernel("aymuos15/rms-norm", version=1)
# call the kernel's registered op, e.g. kernel.<op>(...)
```

> Note: the published build targets **aarch64 / CUDA / sm_121** (NVIDIA GB10). Other platforms need the full build matrix.
