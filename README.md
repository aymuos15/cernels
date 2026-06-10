# kernels

Benchmark custom CUDA kernels against native torch (`op_eager` / `op_compile` / `hub` / `lib` / `custom`), plus tooling to catalog the [HF Kernel Hub](https://huggingface.co/models?filter=kernels).

## Setup

```bash
uv sync
echo "HF_TOKEN=hf_..." > secrets.env
```

## Use

```bash
# benchmark one config — RUN ON THE SPARK, not locally (see AGENTS.md)
ssh spark 'bash -lc "cd ~/kernels && uv run --no-sync python -m benchmark.main sam_decomposed_rel_pos"'
rsync spark:kernels/analysis/ analysis/        # pull results back
uv run --no-sync python -m benchmark.view          # summarize (local ok)
uv run --no-sync python -m modeling.view           # whole-model results (local ok)
```

Runs offline from the HF cache; `HF_TOKEN` is auto-loaded from `secrets.env`. Add `HF_HUB_OFFLINE=0` to fetch an uncached kernel. To add one, follow [`skills/implement-kernel`](skills/implement-kernel/SKILL.md).

Engineering standards — the one-slug-per-kernel naming invariant, no dead code, comment discipline — live in [`RULES.md`](RULES.md); `scripts/check_naming.py` enforces the naming part in pre-commit.

## Results

`uv run --no-sync python -m benchmark.view` over the saved `analysis/` runs (all on GB10 / aarch64, timed with `torch.utils.benchmark` `blocked_autorange` — see [timing methodology](docs/guide/running_benchmarks.md#timing-methodology)). Reference bar is `op_compile`; speedups are `custom` vs that bar.

| config | op_eager(ms) | op_compile(ms) | hub(ms) | lib(ms) | custom(ms) | hub vs ref | custom vs ref | custom ✓ |
|---|---|---|---|---|---|---|---|---|
| multi_scale_deformable_attention | 0.628 | 0.886 | 0.033 | - | 0.023 | 27.09x | 39.19x | ✓ |
| gaussian_blur_2d | 5.521 | 10.808 | - | - | 2.090 | - | 5.17x | ✓ |
| megablocks_moe | - | - | 8.234 | - | 6.326 | - | 1.30x | ✓ |
| non_maximum_suppression | 0.695 | 0.710 | - | - | 0.288 | - | 2.46x | ✓ |
| primus_3d_rope | 2.912 | 0.470 | - | - | 0.441 | - | 1.07x | ✓ |
| rms_norm | 0.146 | 0.160 | - | - | 0.130 | - | 1.23x | ✓ |
| roi_align | 0.086 | 0.106 | - | - | 0.077 | - | 1.39x | ✓ |
| rotary_embedding | 3.032 | 0.619 | 1.207 | - | 0.585 | 0.51x | 1.06x | ✓ |
| sam_decomposed_rel_pos | 4.869 | 4.907 | - | - | 0.208 | - | 23.58x | ✓ |
| silu_and_mul | 1.029 | 0.611 | - | - | 0.589 | - | 1.04x | ✓ |
| gpt_oss_moe_experts | 22.071 | 20.070 | - | - | 15.818 | - | 1.27x | ✓ |
| qwen3_next_moe_experts | 74.629 | 76.189 | - | - | 27.692 | - | 2.75x | ✓ |
| qwen3_next_gated_deltanet | 17.893 | 12.742 | - | - | 9.266 | - | 1.38x | ✓ |
| qwen3_next_gated_rmsnorm | 4.852 | 0.462 | - | - | 0.439 | - | 1.05x | ✓ |
| cohere2_moe_experts | 21.269 | 21.085 | - | - | 9.990 | - | 2.11x | ✓ |
| cohere2_moe_experts_decode | 1.194 | 1.222 | - | - | 0.341 | - | 3.58x | ✓ |

## Whole model — North Mini Code (Cohere2Moe 30B-A3B)

The model-level showcase: [`CohereLabs/North-Mini-Code-1.0`](https://huggingface.co/CohereLabs/North-Mini-Code-1.0) (real weights, 57 GB bf16 on one GB10) with the two `cohere2_moe_experts` kernel entry points swapped into every `Cohere2MoeExperts` instance — grouped GEMM at prefill shapes, fused gather-GEMV at decode shapes. `uv run --no-sync python -m modeling.main north_mini_code` (see [`src/modeling/`](src/modeling/README.md)); prefill 2167 tokens, greedy decode 128.

| variant | prefill ms | decode tok/s | greedy match vs stock |
|---|---|---|---|
| stock (eager) | 826 | 14.2 | — |
| custom: prefill kernel only | 770 (1.07×) | 14.1 | 51/64 |
| custom: prefill + decode kernels | 773 (1.07×) | **23.6 (1.66×)** | **64/64** |

The decode win is the headline: the fused gather-GEMV (3.58× at the op) carries through to **1.66× end-to-end tokens/sec**, with the kernelized model reproducing stock's greedy output exactly for 64 tokens. The prefill gain is honest but small: in-model, transformers 5.10 dispatches experts through its own grouped-mm implementation with the model's real (skewed) routing, which is already much faster per layer than the standalone uniform-routing reference the op table benchmarks against — so the op-level 2.11× compresses to 1.07× end-to-end. The 51/64 on the prefill-only variant is accumulated bf16 drift flipping a near-tie token mid-sequence, not an op-level mismatch (the kernel verifies at the op level; the full-pipeline variant matches 64/64).

## Layout

| path | what |
|---|---|
| [`src/configs/`](src/configs/README.md) | one `Config` per kernel (`registry/`) describing how to benchmark it |
| [`src/kops/`](src/kops/README.md) | the custom kernels (kernel-builder repos) |
| [`src/benchmark/`](src/benchmark/README.md) | run, save, and summarize benchmarks |
| [`src/modeling/`](src/modeling/README.md) | swap kops kernels into a whole model and benchmark stock vs kernelized |
| [`src/profiling/`](src/profiling/README.md) | profile a whole model (prefill + decode) to find kernel-worthy ops |
| [`src/dataset/`](src/dataset/README.md) | build and browse the Hub kernel catalog |
