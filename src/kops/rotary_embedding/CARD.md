---
tags:
- kernel
library_name: kernels
license: apache-2.0
---

# Rotary position embedding

Rotary position embedding (rotate_half convention), matching Llama `apply_rotary_pos_emb`.

Built with [kernel-builder](https://github.com/huggingface/kernel-builder) as a native `torch.library` op (AOT, loadable via the [`kernels`](https://github.com/huggingface/kernels) library). Reference op: transformers `apply_rotary_pos_emb` (Llama).

## Benchmark (NVIDIA GB10 / sm_121, aarch64, bf16)

Mean latency per workload — `op_eager` / `op_compile` are the torch reference (eager and `torch.compile`); `custom` is this kernel. The custom kernel runs at **0.559 ms** — **1.06×** vs `op_compile`, verified against the reference.

| workload | mean (ms) |
|---|---|
| `op_eager` (reference) | 2.888 |
| `op_compile` | 0.595 |
| `hub` | 1.139 |
| `custom` | 0.559 |

## Usage

```python
import torch
from kernels import get_kernel

kernel = get_kernel("aymuos15/rotary-embedding", version=1)
# call the kernel's registered op, e.g. kernel.<op>(...)
```

> Note: the published build targets **aarch64 / CUDA / sm_121** (NVIDIA GB10). Other platforms need the full build matrix.
