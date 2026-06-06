# kernels

Tooling for the [HF Kernel Hub](https://huggingface.co/models?filter=kernels): catalog the published kernels, then benchmark them against native torch.

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
uv run --no-sync python -m benchmark.main relu   # -> analysis/relu/benchmark.{json,log}
uv run --no-sync python -m benchmark.view        # summarize saved results
```

Benchmarks run offline from the HF cache by default (no Hub requests); `HF_TOKEN` is auto-loaded from `secrets.env`. To download a not-yet-cached kernel, run once with `HF_HUB_OFFLINE=0`.

## Layout

| path | what |
|---|---|
| [`src/dataset/`](src/dataset/README.md) | build (`build.py`) and browse (`view.py`) the kernel catalog |
| [`src/configs/`](src/configs/README.md) | one `Config` subclass per kernel (in `registry/`) describing how to benchmark it |
| [`src/kops/`](src/kops/README.md) | custom kernels (your own), benchmarked head-to-head via a config's `custom:` |
| [`src/benchmark/`](src/benchmark/README.md) | `main.py` runs a benchmark, `monitor.py` saves it to `analysis/`, `view.py` summarizes |
