# Running benchmarks

## Where to run — the Spark only

**Never run benchmarks or build kernels on the local machine** (see [AGENTS.md](../../AGENTS.md)). Everything runs on the **Spark** (GB10 / Grace-Blackwell, `ssh sie271-pc`): transfer with `scripts/transfer.sh sie271-pc`, then run over ssh, then pull `analysis/` back.

```bash
scripts/transfer.sh sie271-pc                                                    # sync repo to the Spark
ssh sie271-pc 'bash -lc "cd ~/kernels && uv run --no-sync python -m benchmark.main <name>"'
rsync sie271-pc:kernels/analysis/ analysis/                                       # pull results back
uv run --no-sync python -m benchmark.view                                        # view locally (read-only)
```

`<name>` = a `Config.name` in `src/configs/registry/`. Use `uv run` (not `.venv/bin/python` directly): a custom CUDA kernel built with `load_inline` needs `ninja` on `PATH`, and `uv run` puts `.venv/bin` there. Results are written per host under `analysis/<host>/`.

Hub kernels (`lib`) generally **do** run on the Spark: current `kernels-community` repos publish a `torch212-cxx11-cu130-aarch64-linux` build under Version 1 that resolves as "compatible, preferred ✅" on GB10 / sm_121, and our `HubConfig` (`get_kernel(repo, version=1)`) loads it. Check a repo with `kernels versions <repo>` before assuming. Only if no compatible build exists for the Spark's (torch 2.12 / cu130 / aarch64) combo is the `lib` workload **skipped** (shows `·`), with eager / compile / custom still running and verifying.

## Downloading a Hub kernel

Benchmarks run **offline from the HF cache** by default (no Hub requests); `HF_TOKEN` auto-loads from `secrets.env`. To fetch a not-yet-cached kernel once:

```bash
HF_HUB_OFFLINE=0 uv run --no-sync python -m benchmark.main <name>
```

## Reading the result

`view.py` columns: `eager / compile / lib / custom` (mean ms), `lib vs compile` / `custom vs compile` (speedups), and `lib ✓` / `custom ✓` (verification — `✓` pass, `✗` fail, `·` ran but no verdict / skipped, `-` not defined for this config).

**Success = the contender (`lib` or `custom`) verifies `✓` and beats `compile`.** `torch.compile` already saturates memory bandwidth on elementwise ops, so the bar is *compile*, not eager — a kernel that only beats eager isn't a win.

The full record is saved to `analysis/<host>/<name>.json` (means, std, speedups, `verified` flags, provenance) plus a `.log`.
