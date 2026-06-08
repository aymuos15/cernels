---
tags:
- kernel
library_name: kernels
license: apache-2.0
---

# Non-maximum suppression

Greedy non-maximum suppression over boxes by IoU, matching `torchvision.ops.nms`.

Built with [kernel-builder](https://github.com/huggingface/kernel-builder) as a native `torch.library` op (AOT, loadable via the [`kernels`](https://github.com/huggingface/kernels) library). Reference op: torchvision `ops.nms`.

## Benchmark (NVIDIA GB10 / sm_121, aarch64, bf16)

Mean latency per workload — `op_eager` / `op_compile` are the torch reference (eager and `torch.compile`); `custom` is this kernel. The custom kernel runs at **0.288 ms** — **2.53×** vs `op_compile`, verified against the reference.

| workload | mean (ms) |
|---|---|
| `op_eager` (reference) | 0.711 |
| `op_compile` | 0.729 |
| `custom` | 0.288 |

## Usage

```python
import torch
from kernels import get_kernel

kernel = get_kernel("aymuos15/non-maximum-suppression", version=1)
# call the kernel's registered op, e.g. kernel.<op>(...)
```

> Note: the published build targets **aarch64 / CUDA / sm_121** (NVIDIA GB10). Other platforms need the full build matrix.
