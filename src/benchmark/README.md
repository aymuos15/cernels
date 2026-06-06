# benchmark

Benchmark a Hub kernel against native torch: **eager vs compile vs kernel** (with a
correctness check), via the `kernels` benchmark harness.

```bash
uv run --no-sync python src/benchmark/main.py <config>   # config = a yaml in src/configs/
uv run --no-sync python src/benchmark/view.py            # summarize saved results
```

Each kernel is described by a YAML in [`src/configs/`](../configs/README.md). `main.py`
loads it, verifies the kernel matches the `baseline`, then times all three.

- `main.py` — run one benchmark
- `monitor.py` — captures the run and writes `analysis/<config>/benchmark.{json,log}`
- `view.py` — reads `analysis/` and prints a summary table
