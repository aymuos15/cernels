---
name: implement-kernel
description: Step-by-step workflow to implement and benchmark one kernel + baseline from a .issues backlog item (eager vs compile vs lib vs custom).
---

# Implement a kernel (one issue at a time)

Take a single backlog issue from `.issues/` to a benchmarked, verified result. Each step links the guide with the detail — read the linked guide, don't guess.

> **Never run benchmarks or build kernels locally — only on the Spark** (`ssh sie271-pc`). See [AGENTS.md](../../AGENTS.md).

## 0. Pick the issue
The user points you to `.issues/<n>-<name>.md`. Read it **and** its row in [`.issues/list.md`](../../.issues/list.md) — the row gives the baseline reference, whether there's a Hub `lib`, whether a `custom` kernel is expected, and any hardware note.

## 1. Set up the environment / download
All runs happen on the **Spark** (`ssh sie271-pc`) — `scripts/transfer.sh sie271-pc` to sync. See [running benchmarks](../../docs/guide/running_benchmarks.md).
- **Hub `lib` (a `kernels-community/...` repo):** make sure it's cached on the Spark — run once with `HF_HUB_OFFLINE=0`. Hub kernels have **no GB10 / sm_121 build**, so the `lib` workload is skipped there (`·`); eager / compile / custom still run.
- **Write-our-own:** open the cited reference implementation (the link in the list.md baseline column) and read it before reproducing it.

## 2. Set up the baseline
Follow [setting up baselines](../../docs/guide/setting_up_baselines.md): built-in torch op → else mirror existing repo code → else write it in torch. Then choose `verify()` and `use_compile` per [correctness](../../docs/guide/correctness.md).

## 3. Write the config
[How to add a config](../../docs/guide/how_to_add_a_config.md): a plain `Config` (non-Hub: baseline + custom) or `HubConfig` (`lib` is a Hub kernel). Set `name` / `dtype` / `op` / `inputs` / `baseline` / `verify`, then register it in `CONFIGS` (`src/configs/registry/__init__.py`).

## 4. Write the custom kernel (if the issue has `custom ✓`)
[How to add a custom kernel](../../docs/guide/how_to_add_a_custom_kernel.md): add `src/kops/registry/<name>.{py,cu}`, expose `kernel(*inputs)`, and wire `custom = staticmethod(kernel)` in the config.

## 5. Benchmark and read the result
On the Spark: `ssh sie271-pc 'bash -lc "cd ~/kernels && uv run --no-sync python -m benchmark.main <name>"'`, then pull `analysis/` back and `uv run --no-sync python -m benchmark.view` locally. **Success = the contender (`lib`/`custom`) verifies ✓ AND beats `torch.compile`** — compile already saturates bandwidth, so the bar is compile, not eager. The record lands in `analysis/<host>/<name>.json`. See [running benchmarks](../../docs/guide/running_benchmarks.md) for the column meanings.

**Done when:** the config runs, every present non-eager workload verifies ✓, and you've reported the speedup vs compile.
