---
tags:
- kernel
library_name: kernels
license: apache-2.0
---

# Qwen3-Next fused gated RMSNorm

Fused gated RMSNorm (`Qwen3NextRMSNormGated`): RMS-normalize, scale by weight, then multiply by `silu(gate)` — all in one pass, reduction in fp32.

Built with [kernel-builder](https://github.com/huggingface/kernel-builder) as a native `torch.library` op (AOT, loadable via the [`kernels`](https://github.com/huggingface/kernels) library). Reference op: transformers `Qwen3NextRMSNormGated`.

## Benchmark (NVIDIA GB10 / sm_121, aarch64, bf16)

Mean latency per workload — `op_eager` / `op_compile` are the torch reference (eager and `torch.compile`); `custom` is this kernel. The custom kernel runs at **0.471 ms** — **1.02×** vs `op_compile` (honest parity — this op is bandwidth-bound and `torch.compile` already saturates it).

| workload | mean (ms) |
|---|---|
| `op_eager` (reference) | 5.232 |
| `op_compile` | 0.482 |
| `custom` | 0.471 |

## Usage

```python
import torch
from kernels import get_kernel

kernel = get_kernel("aymuos15/qwen3-next-gated-rmsnorm", version=1)
# call the kernel's registered op, e.g. kernel.<op>(...)
```

> Note: the published build targets **aarch64 / CUDA / sm_121** (NVIDIA GB10). Other platforms need the full build matrix.
