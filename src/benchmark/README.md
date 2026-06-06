# benchmark

Benchmark an op against native torch: **eager vs compile vs lib vs custom**, each verified against the `baseline`.

```bash
uv run --no-sync python -m benchmark.main <name>     # name = a Config in src/configs/registry/
uv run --no-sync python -m benchmark.view            # summarize saved results
```

Runs offline from the HF cache by default (`HF_TOKEN` auto-loaded from `secrets.env`); use `HF_HUB_OFFLINE=0` to download a not-yet-cached Hub kernel.

Each op is described by a `Config`/`HubConfig` subclass in [`src/configs/`](../configs/README.md). `main.py` loads it, runs each present workload (eager / compile / lib / custom), verifies each against the `baseline`, and saves the result.

- `main.py` — run one benchmark (the `run()` workload loop)
- `monitor.py` — measurement: capture run output, `time_ms` (one CUDA-timed call), `stats` (reduce timings)
- `save.py` — writes `analysis/<host>/<config>.{json,log}` (latest run per machine)
- `view.py` — reads `analysis/` and prints a summary table
