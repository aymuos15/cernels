---
tags:
- kernel
library_name: kernels
license: apache-2.0
---

# Qwen3-Next Gated DeltaNet

Chunked delta-rule linear attention (Gated DeltaNet) — l2-normalized q/k, gated cumulative decay, an `(I−A)⁻¹` within-chunk solve, and an fp32 cross-chunk state kept in shared memory. The signature linear-attention mixer of Qwen3-Next / Qwen3.5.

Built with [kernel-builder](https://github.com/huggingface/kernel-builder) as a native `torch.library` op (AOT, loadable via the [`kernels`](https://github.com/huggingface/kernels) library). Reference op: transformers `torch_chunk_gated_delta_rule` (Qwen3-Next).

## Benchmark (NVIDIA GB10 / sm_121, aarch64, bf16)

Mean latency per workload — `op_eager` / `op_compile` are the torch reference (eager and `torch.compile`); `custom` is this kernel. The custom kernel runs at **9.286 ms** — **1.34×** vs `op_compile`, verified against the reference.

| workload | mean (ms) |
|---|---|
| `op_eager` (reference) | 18.120 |
| `op_compile` | 12.460 |
| `custom` | 9.286 |

## Usage

```python
import torch
from kernels import get_kernel

kernel = get_kernel("aymuos15/qwen3-next-gated-deltanet", version=1)
# call the kernel's registered op, e.g. kernel.<op>(...)
```

> Note: the published build targets **aarch64 / CUDA / sm_121** (NVIDIA GB10). Other platforms need the full build matrix.
