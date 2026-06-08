# profiling

End-to-end model profiler — find where a whole model spends its time, so we know which custom kernel to write. Reusable across architectures (the analog of `benchmark/` + `configs/`, but for whole models instead of single ops).

```bash
# RUN ON THE SPARK, not here (see AGENTS.md):
ssh spark 'bash -lc "cd ~/kernels && HF_HUB_OFFLINE=0 uv run --no-sync python -m profiling.main <model>"'
rsync spark:kernels/analysis/ analysis/        # pull results back
uv run --no-sync python -m profiling.view          # render saved profiles (read-only, local ok)
```

`<model>` = a `ModelProfile.name` in [`registry/`](registry/). Add a model = one subclass there (define `load()` + `inputs()`), exactly like adding a kernel config.

## What it measures

Two phases, profiled separately (their op-mix differs completely):

- **prefill** — the first forward (for a VLM this includes the vision encode).
- **decode** — `decode_tokens` incremental steps from the KV cache (the per-token latency that usually matters).

Three lenses per phase:

| lens | how | answers |
|---|---|---|
| modules | per-module CUDA-event timing → **self** time per class (inclusive − direct children) | which *layer type* dominates (kernelize target); functional work like sdpa/rope shows as its module's self time |
| ops | `torch.profiler` / Kineto, top CUDA ops by self device time | which *CUDA kernel* dominates |
| inductor | `torch.compile` the model + dump generated Triton | what compile already fuses (and a Triton starting point); skip with `--no-inductor` |

> Spark/GB10 is UMA and has profiler quirks (`ncu` counter perms, unreliable memory telemetry, Arm SBSA Nsight). See [profiling on the Spark](../../docs/guide/profiling_on_spark.md) — lead with timeline + tok/s, don't over-trust VRAM numbers.

Output (one dir per model): `analysis/<host>/profile/<model>/` with `profile.json`, `report.txt`, and `inductor/` — the generated Triton per graph region (`<region>.py` + `<region>.graph.py`) plus an `INDEX.md` manifest listing the fused `triton_*` kernel names (which reveal what compile already fuses).
