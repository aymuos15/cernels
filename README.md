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

## Layout

| path | what |
|---|---|
| [`src/configs/`](src/configs/README.md) | one `Config` per kernel (`registry/`) describing how to benchmark it |
| [`src/kops/`](src/kops/README.md) | the custom kernels (kernel-builder repos) |
| [`src/benchmark/`](src/benchmark/README.md) | run, save, and summarize benchmarks |
| [`src/dataset/`](src/dataset/README.md) | build and browse the Hub kernel catalog |
