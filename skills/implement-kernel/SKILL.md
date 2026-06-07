---
name: implement-kernel
description: Full operating manual to implement and benchmark one kernel + reference from a .issues backlog item (op_eager vs op_compile vs hub vs lib vs custom). Launch an agent with just the issue number; everything else is here.
---

# Implement a kernel (one issue at a time)

Take a single backlog issue from `.issues/kernel/` to a benchmarked, verified result. **Read this whole file and every guide it links before writing code — don't guess.**

> **Never run benchmarks or build kernels locally — only on the Spark** (`ssh sie271-pc`). See [AGENTS.md](../../AGENTS.md).

## Launching (for whoever starts the agent)
Two file mentions are the whole invocation: **`@.issues/kernel/<N>-<name>.md  @skills/implement-kernel/SKILL.md`** — no instruction text needed. The issue file is the *what*, this skill is the *how*. Read everything below, including the linked guides, before writing code; all rules, conventions, and the report contract live in this file.

## Rules
- **Spark-only** for any run or `.cu` build: edit locally → `scripts/transfer.sh sie271-pc` → run over ssh → `rsync sie271-pc:kernels/analysis/ analysis/` to pull results back. The Spark uses a password — if auth hangs, note it and stop; don't retry forever.
- **Don't `git commit`/`push`.** Leave changes in the working tree for review.
- **Touch only this issue's files**: its config, its `kops` kernel, and the one `CONFIGS` registry line. Don't edit other issues' files or the `.issues/kernel/` docs.
- **Keep the op's distinctive work in the timed path.** `inputs()` builds raw inputs only — never precompute the op-specific transform there (e.g. don't pre-build interleaved cos/sin in `inputs()`), or you benchmark a trivialized op and the "win" is hollow. See [correctness](../../docs/guide/correctness.md).

## Conventions
- **Config name** = the issue slug with hyphens as underscores (e.g. `dual-rope` → `dual_rope`); it's the CLI arg to `benchmark.main`.
- **Config** → `src/configs/registry/<name>.py`; add its class to `CONFIGS` in `src/configs/registry/__init__.py`.
- **Custom kernel** (if the issue row shows `custom ✓`) → a **kernel-builder repo** at `src/kops/<name>/` (build.toml + csrc/ + torch-ext/ + flake.nix), plus a thin loader `src/kops/registry/<name>.py` exposing `kernel(*inputs)`, wired as `custom = staticmethod(kernel)`. Built AOT via nix on the Spark; NOT load_inline.
- **Study these first**: `src/configs/registry/rotary.py`, `nms.py`; `src/kops/rmsnorm/` and `src/kops/rope/` (kernel-builder repos); `src/kops/registry/rmsnorm.py` (loader); `src/configs/base.py`.

## Steps

### 0. Pick the issue
The launcher gives you `<N>`. Read `.issues/kernel/<N>-<name>.md` **and** its row in [`.issues/kernel/list.md`](../../.issues/kernel/list.md) — the row gives the reference op, whether there's a Hub kernel (`hub`), whether a `custom` kernel is expected, and any hardware note.

### 1. Set up the environment / download
All runs are on the Spark; `scripts/transfer.sh sie271-pc` to sync. See [running benchmarks](../../docs/guide/running_benchmarks.md).
- **Hub kernel (`hub`):** make sure it's cached on the Spark (run once with `HF_HUB_OFFLINE=0`). Hub kernels generally **do** run on the Spark — current `kernels-community` repos ship a `torch212-cxx11-cu130-aarch64-linux` Version 1 build that resolves "compatible, preferred ✅" on GB10/sm_121, and `HubConfig` loads it via `get_kernel(repo, version=1)`. Verify with `kernels versions <repo>` first; the `hub` workload is skipped (`·`) only when no build matches the Spark's torch 2.12 / cu130 / aarch64.
- **Library reference op:** open the cited reference function (the link in the issue's reference column) and read it — WebFetch the upstream file if needed — before wiring it in. **Never hand-write a reference** (see [setting up baselines](../../docs/guide/setting_up_baselines.md)); call the real library op directly.

### 2. Set up the baseline
Follow [setting up baselines](../../docs/guide/setting_up_baselines.md): library reference op (`op_eager`/`op_compile`) → else a Hub kernel as the reference (`reference_is_hub`). **Never hand-write a reference** — if neither exists, the op isn't ready; pick another. Then choose `verify()` and `use_compile` per [correctness](../../docs/guide/correctness.md). `inputs(device, dtype)` must also work on the `meta` device.

### 3. Write the config
[How to add a config](../../docs/guide/how_to_add_a_config.md): plain `Config` (non-Hub: `baseline` + `custom`) or `HubConfig` (a Hub `hub` contender). Set `name` / `dtype` / `op` / `inputs` / `baseline` / `verify`, then register in `CONFIGS`.

### 4. Write the custom kernel (if the issue has `custom ✓`)
[How to add a custom kernel](../../docs/guide/how_to_add_a_custom_kernel.md): scaffold a kernel-builder repo `src/kops/<name>/` (build.toml, csrc/<name>.cu with `#include <torch/all.h>`, torch-ext/torch_binding.{cpp,h} registering a native `TORCH_LIBRARY` op, torch-ext/<name>/__init__.py, flake.nix) + a loader `src/kops/registry/<name>.py` (`get_local_kernel`). Hard rules: **`[general] name` uses dashes** (not underscores); **integer op args are `int64_t` in C++** (schema says `int`); build.toml needs `version` + `[general.hub] repo-id`; keep Python prep in the loader, not the op; reduce in fp32. Build on the Spark with `scripts/build_kernels.sh <name>` (nix → `build/`).

### 5. Benchmark and read the result
`ssh sie271-pc 'bash -lc "cd ~/kernels && uv run --no-sync python -m benchmark.main <name>"'`, then pull `analysis/` back and run `uv run --no-sync python -m benchmark.view` locally. See [running benchmarks](../../docs/guide/running_benchmarks.md) for the column meanings.

## Report (definition of done)
Done when the config runs on the Spark and every present contender workload verifies `✓`. Work quietly during the task; return a **concise final report** with exactly:
1. **Files** created / edited.
2. **Benchmark table** — op_eager / op_compile / hub / lib / custom mean ms, the custom-vs-op_compile (and hub/lib-vs-op_compile) speedup, and `✓`/`✗` per workload.
3. **Verdict** — did the contender beat `op_compile`? That's the success bar (compile already saturates bandwidth, so beating op_eager alone is not a win).
4. **Decisions / problems** — anything non-obvious you chose or hit. State explicitly **what runs inside the timed path vs precomputed in `inputs()`**, so a trivialized benchmark can't pass as a win.
