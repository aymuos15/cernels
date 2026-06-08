# Running benchmarks

## Where to run — the Spark only

**Never run benchmarks or build kernels on the local machine** (see [AGENTS.md](../../AGENTS.md)). Everything runs on the **Spark** (GB10 / Grace-Blackwell, `ssh spark`): transfer with `scripts/transfer.sh spark`, then run over ssh, then pull `analysis/` back.

```bash
scripts/transfer.sh spark                                                    # sync repo to the Spark
ssh spark 'bash -lc "cd ~/kernels && bash scripts/build_kernels.sh"'          # build custom kernels (nix)
ssh spark 'bash -lc "cd ~/kernels && uv run --no-sync python -m benchmark.main <name>"'
rsync spark:kernels/analysis/ analysis/                                       # pull results back
uv run --no-sync python -m benchmark.view                                        # view locally (read-only)
```

`<name>` = a `Config.name` in `src/configs/registry/`. **Custom kernels are kernel-builder kernels** (`src/kops/<name>/`): build them on the Spark with `scripts/build_kernels.sh [<name>...]` (nix → `src/kops/<name>/build/`) before benchmarking — the loader `get_local_kernel`s the built variant. `build/` is gitignored and survives `transfer.sh` (excluded), so you only rebuild when a kernel's source changes. Results are written per host under `analysis/<host>/`.

Baseline libraries (`torchvision` for non_maximum_suppression/roi_align, `kornia` for gaussian_blur_2d) are an **optional extra**, not core deps. Install them once on the Spark with `uv sync --extra benchmark`; configs that need them import lazily, so other configs run without the extra.

Hub kernels (the `hub` workload) generally **do** run on the Spark: current `kernels-community` repos publish a `torch212-cxx11-cu130-aarch64-linux` build under Version 1 that resolves as "compatible, preferred ✅" on GB10 / sm_121, and our `HubConfig` (`get_kernel(repo, version=1)`) loads it. Check a repo with `kernels versions <repo>` before assuming. Only if no compatible build exists for the Spark's (torch 2.12 / cu130 / aarch64) combo is the `hub` workload **skipped** (shows `·`), with op_eager / op_compile / custom still running and verifying.

## Downloading a Hub kernel

Benchmarks run **offline from the HF cache** by default (no Hub requests); `HF_TOKEN` auto-loads from `secrets.env`. To fetch a not-yet-cached kernel once:

```bash
HF_HUB_OFFLINE=0 uv run --no-sync python -m benchmark.main <name>
```

## Reading the result

`view.py` columns: `op_eager / op_compile / hub / lib / custom` (mean ms), `hub|lib|custom vs ref` (speedups vs the reference bar = `op_compile`, else `op_eager`, else `hub`), and `hub|lib|custom ✓` (verification — `✓` pass, `✗` fail, `·` ran but no verdict / skipped, `-` not defined for this config). The reference's own column shows `-` for speedup/verify.

**Success = the contender (`hub` / `lib` / `custom`) verifies `✓` and beats `op_compile`.** `torch.compile` already saturates memory bandwidth on elementwise ops, so the bar is *op_compile*, not op_eager — a kernel that only beats eager isn't a win.

The full record is saved to `analysis/<host>/<name>.json` (means, std, speedups, `verified` flags, provenance) plus a `.log`.
