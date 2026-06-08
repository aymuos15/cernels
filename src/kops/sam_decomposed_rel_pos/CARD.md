---
tags:
- kernel
library_name: kernels
license: apache-2.0
---

# SAM decomposed relative-position bias

SAM's decomposed relative-position attention bias (`add_decomposed_rel_pos`): interpolated `get_rel_pos` tables, two relative-position einsums, and the broadcast add into pre-softmax logits, fused.

Built with [kernel-builder](https://github.com/huggingface/kernel-builder) as a native `torch.library` op (AOT, loadable via the [`kernels`](https://github.com/huggingface/kernels) library). Reference op: transformers `add_decomposed_rel_pos` (SAM).

## Benchmark (NVIDIA GB10 / sm_121, aarch64, bf16)

Mean latency per workload — `op_eager` / `op_compile` are the torch reference (eager and `torch.compile`); `custom` is this kernel. The custom kernel runs at **0.227 ms** — **21.61×** vs `op_compile`, verified against the reference.

| workload | mean (ms) |
|---|---|
| `op_eager` (reference) | 4.886 |
| `op_compile` | 4.914 |
| `custom` | 0.227 |

## Usage

```python
import torch
from kernels import get_kernel

kernel = get_kernel("aymuos15/sam-decomposed-rel-pos", version=1)
# call the kernel's registered op, e.g. kernel.<op>(...)
```

> Note: the published build targets **aarch64 / CUDA / sm_121** (NVIDIA GB10). Other platforms need the full build matrix.
