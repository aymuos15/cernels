---
tags:
- kernel
library_name: kernels
license: apache-2.0
---

# SAM decomposed relative-position bias

SAM's decomposed relative-position attention bias (`add_decomposed_rel_pos`): one block per query row stages the query in shared memory, computes the k_h + k_w rel-pos dot products once per row (GEMM-equivalent traffic, fp32 accumulation), and fuses the broadcast bias add into the same launch. With `attn=None` it writes the SDPA `attn_mask` bias directly — no zeros materialization.

Built with [kernel-builder](https://github.com/huggingface/kernel-builder) as a native `torch.library` op (AOT, loadable via the [`kernels`](https://github.com/huggingface/kernels) library). Reference op: transformers `add_decomposed_rel_pos` (SAM).

## Benchmark (NVIDIA GB10 / sm_121, aarch64)

Mean latency per workload — `op_eager` / `op_compile` are the torch reference (eager and `torch.compile`); `custom` is this kernel. Two shapes, because they tell different stories: the single-window shape (B = 12 heads, window 14, fp16) is launch-bound and fusion wins big; the in-model shape (B = 600 — 25 windows × 12 heads × 2 encoder views, the batch DeepSeek-OCR-2's window partitioning actually passes, bf16) is where the eager einsums become efficient batched GEMMs, and the kernel still wins **6.4×** by computing each rel dot once per row and writing the bias in one pass.

| shape | op_eager (ms) | op_compile (ms) | custom (ms) | custom vs op_compile |
|---|---|---|---|---|
| B=12, window 14, fp16 | 4.857 | 4.896 | 0.070 | **69.6×** |
| B=600, window 14, bf16 | 5.135 | 5.513 | 0.858 | **6.4×** |

In-model (DeepSeek-OCR-2, `modeling.main deepseek_ocr_2`), the swap feeds the bias straight to `F.scaled_dot_product_attention` as `attn_mask` for both the windowed (14×14 keys) and global (64×64 keys) SAM layers: prefill goes from 315 ms (stock) / 312 ms (MoE kernels only) to **274 ms** (1.15× vs stock), 64/64 greedy-token match — where the previous per-output-element version of this kernel regressed to 407 ms.

## Usage

```python
import torch
from kernels import get_kernel

kernel = get_kernel("aymuos15/sam-decomposed-rel-pos", version=1)
# bias-add mode (eager attention path):
out = kernel.sam_decomposed_rel_pos(query, Rh, Rw, attn)
# bias-only mode (SDPA attn_mask path):
bias = kernel.sam_decomposed_rel_pos(query, Rh, Rw)
```

> Note: the published build targets **aarch64 / CUDA / sm_121** (NVIDIA GB10). Other platforms need the full build matrix.
