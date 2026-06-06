# kernels

Tooling for the [HF Kernel Hub](https://huggingface.co/models?filter=kernels): catalog the
published kernels, then benchmark them against native torch.

## Setup

```bash
uv sync                       # install deps + the local packages
echo "HF_TOKEN=hf_..." > secrets.env
```

## Use

```bash
# dataset — build & browse the catalog of Hub kernels
uv run --no-sync python src/dataset/build.py    # -> kernels.parquet
uv run --no-sync python src/dataset/view.py     # interactive browser

# benchmark — eager vs compile vs kernel for one kernel
uv run --no-sync python src/benchmark/main.py relu   # -> analysis/relu/benchmark.{json,log}
uv run --no-sync python src/benchmark/view.py        # summarize saved results
```

## Layout

| path | what |
|---|---|
| [`src/dataset/`](src/dataset/README.md) | build (`build.py`) and browse (`view.py`) the kernel catalog |
| [`src/configs/`](src/configs/README.md) | one YAML per kernel to benchmark, plus `helpers.py` |
| [`src/benchmark/`](src/benchmark/README.md) | `main.py` runs a benchmark, `monitor.py` saves it to `analysis/`, `view.py` summarizes |
