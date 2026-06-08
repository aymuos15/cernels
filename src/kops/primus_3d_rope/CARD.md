---
tags:
- kernel
library_name: kernels
license: apache-2.0
---

# Axial 3D rotary embedding

Axial 3D rotary position embedding (interleaved convention) as used by Primus / timm `apply_rot_embed_cat`.

Built with [kernel-builder](https://github.com/huggingface/kernel-builder) as a native `torch.library` op (AOT, loadable via the [`kernels`](https://github.com/huggingface/kernels) library). Reference op: timm `RotaryEmbeddingCat` / `apply_rot_embed_cat` (Primus).

## Benchmark (NVIDIA GB10 / sm_121, aarch64, bf16)

Mean latency per workload — `op_eager` / `op_compile` are the torch reference (eager and `torch.compile`); `custom` is this kernel. The custom kernel runs at **0.418 ms** — **1.06×** vs `op_compile`, verified against the reference.

| workload | mean (ms) |
|---|---|
| `op_eager` (reference) | 2.647 |
| `op_compile` | 0.442 |
| `custom` | 0.418 |

## Usage

```python
import torch
from kernels import get_kernel

kernel = get_kernel("aymuos15/primus-3d-rope", version=1)
# call the kernel's registered op, e.g. kernel.<op>(...)
```

> Note: the published build targets **aarch64 / CUDA / sm_121** (NVIDIA GB10). Other platforms need the full build matrix.
