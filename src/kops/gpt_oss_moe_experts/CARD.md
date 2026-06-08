---
tags:
- kernel
library_name: kernels
license: apache-2.0
---

# gpt-oss MoE experts

Fused grouped-GEMM MoE for OpenAI gpt-oss: per-expert gate_up/down GEMMs plus the gpt-oss clamped/limited interleaved SwiGLU (`clamp(max=7)`/`clamp(±7)`, `alpha=1.702`, `(up+1)*glu`). 128/32 experts, top-4.

Built with [kernel-builder](https://github.com/huggingface/kernel-builder) as a native `torch.library` op (AOT, loadable via the [`kernels`](https://github.com/huggingface/kernels) library). Reference op: transformers `GptOssExperts` / `GptOssMLP`.

## Benchmark (NVIDIA GB10 / sm_121, aarch64, bf16)

Mean latency per workload — `op_eager` / `op_compile` are the torch reference (eager and `torch.compile`); `custom` is this kernel. The custom kernel runs at **15.545 ms** — **1.28×** vs `op_compile`, verified against the reference.

| workload | mean (ms) |
|---|---|
| `op_eager` (reference) | 21.870 |
| `op_compile` | 19.882 |
| `custom` | 15.545 |

## Usage

```python
import torch
from kernels import get_kernel

kernel = get_kernel("aymuos15/gpt-oss-moe-experts", version=1)
# call the kernel's registered op, e.g. kernel.<op>(...)
```

> Note: the published build targets **aarch64 / CUDA / sm_121** (NVIDIA GB10). Other platforms need the full build matrix.
