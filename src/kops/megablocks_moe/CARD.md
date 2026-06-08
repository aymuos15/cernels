---
tags:
- kernel
library_name: kernels
license: apache-2.0
---

# MegaBlocks-style grouped-GEMM MoE

Fused cuBLAS Tensor-Core grouped-GEMM MoE (token permute, per-expert GEMM, scatter-combine), benchmarked against the MegaBlocks Hub kernel as reference.

Built with [kernel-builder](https://github.com/huggingface/kernel-builder) as a native `torch.library` op (AOT, loadable via the [`kernels`](https://github.com/huggingface/kernels) library). Reference op: kernels-community/megablocks MoE.

## Benchmark (NVIDIA GB10 / sm_121, aarch64, bf16)

Mean latency per workload — `op_eager` / `op_compile` are the torch reference (eager and `torch.compile`); `custom` is this kernel. The custom kernel runs at **6.446 ms** — **1.26×** vs `hub`, verified against the reference.

| workload | mean (ms) |
|---|---|
| `hub` (reference) | 8.124 |
| `custom` | 6.446 |

## Usage

```python
import torch
from kernels import get_kernel

kernel = get_kernel("aymuos15/megablocks-moe", version=1)
# call the kernel's registered op, e.g. kernel.<op>(...)
```

> Note: the published build targets **aarch64 / CUDA / sm_121** (NVIDIA GB10). Other platforms need the full build matrix.
