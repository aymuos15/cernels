# benchmark

Benchmark a Hub kernel against native torch: **eager vs compile vs kernel** (with a correctness check), via the `kernels` benchmark harness.

```bash
uv run --no-sync python -m benchmark.main <name>     # name = a Config in src/configs/registry/
uv run --no-sync python -m benchmark.view            # summarize saved results
```

Runs offline from the HF cache by default (`HF_TOKEN` auto-loaded from `secrets.env`); use `HF_HUB_OFFLINE=0` to download a not-yet-cached kernel.

Each kernel is described by a `Config` subclass in [`src/configs/`](../configs/README.md). `main.py` loads it, verifies the kernel matches the `baseline`, then times all three.

- `main.py` — run one benchmark
- `monitor.py` — captures the run and writes `analysis/<config>/benchmark.{json,log}`
- `view.py` — reads `analysis/` and prints a summary table
