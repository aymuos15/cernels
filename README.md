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
```

Runs offline from the HF cache; `HF_TOKEN` is auto-loaded from `secrets.env`. Add `HF_HUB_OFFLINE=0` to fetch an uncached kernel. To add one, follow [`skills/implement-kernel`](skills/implement-kernel/SKILL.md).

Engineering standards — the one-slug-per-kernel naming invariant, no dead code, comment discipline — live in [`RULES.md`](RULES.md); `scripts/check_naming.py` enforces the naming part in pre-commit.

## Results

`uv run --no-sync python -m benchmark.view` over the saved `analysis/` runs (all on GB10 / aarch64). Reference bar is `op_compile`; speedups are `custom` vs that bar.

| config | op_eager(ms) | op_compile(ms) | hub(ms) | lib(ms) | custom(ms) | hub vs ref | custom vs ref | custom ✓ |
|---|---|---|---|---|---|---|---|---|
| multi_scale_deformable_attention | 0.637 | 0.904 | 0.036 | - | 0.026 | 24.76x | 34.90x | ✓ |
| gaussian_blur_2d | 5.528 | 10.139 | - | - | 2.085 | - | 4.86x | ✓ |
| megablocks_moe | - | - | 8.124 | - | 6.446 | - | 1.26x | ✓ |
| non_maximum_suppression | 0.711 | 0.729 | - | - | 0.288 | - | 2.53x | ✓ |
| primus_3d_rope | 2.647 | 0.442 | - | - | 0.418 | - | 1.06x | ✓ |
| rms_norm | 0.147 | 0.157 | - | - | 0.128 | - | 1.23x | ✓ |
| roi_align | 0.093 | 0.107 | - | - | 0.081 | - | 1.32x | ✓ |
| rotary_embedding | 2.888 | 0.595 | 1.139 | - | 0.559 | 0.52x | 1.06x | ✓ |
| sam_decomposed_rel_pos | 4.886 | 4.914 | - | - | 0.227 | - | 21.61x | ✓ |
| silu_and_mul | 0.978 | 0.590 | - | - | 0.596 | - | 0.99x | ✓ |
| gpt_oss_moe_experts | 21.87 | 19.882 | - | - | 15.545 | - | 1.28x | ✓ |
| qwen3_next_moe_experts | 74.609 | 75.398 | - | - | 27.930 | - | 2.70x | ✓ |
| qwen3_next_gated_deltanet | 17.95 | 12.48 | - | - | 9.27 | - | 1.35x | ✓ |
| qwen3_next_gated_rmsnorm | 5.232 | 0.482 | - | - | 0.471 | - | 1.02x | ✓ |

## Layout

| path | what |
|---|---|
| [`src/configs/`](src/configs/README.md) | one `Config` per kernel (`registry/`) describing how to benchmark it |
| [`src/kops/`](src/kops/README.md) | the custom kernels (kernel-builder repos) |
| [`src/benchmark/`](src/benchmark/README.md) | run, save, and summarize benchmarks |
| [`src/dataset/`](src/dataset/README.md) | build and browse the Hub kernel catalog |
