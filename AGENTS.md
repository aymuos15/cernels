# Agent instructions

## Never run benchmarks or kernels on this machine — only on the Spark

All benchmarking, CUDA builds (`load_inline`), and kernel runs happen on the **Spark** (the GB10 / Grace-Blackwell box, `ssh sie271-pc`), never on the local machine. Transfer the repo with `scripts/transfer.sh sie271-pc`, then run over ssh:

```bash
ssh sie271-pc 'bash -lc "cd ~/kernels && uv run --no-sync python -m benchmark.main <name>"'
```

Do not run `benchmark.main` / `benchmark.view` or compile a `.cu` locally. Pull results back for viewing with `rsync sie271-pc:kernels/analysis/ analysis/`.

Hub kernels (the `hub` workload) generally **do** build on the Spark — current `kernels-community` repos ship a Version 1 `...cu130-aarch64-linux` build that resolves on GB10 / sm_121. Only when no compatible build exists is the `hub` workload **skipped** (shows `·`); op_eager / op_compile / custom still run and verify. See [docs/guide/running_benchmarks.md](docs/guide/running_benchmarks.md).

## Workflow

To scout the HF + inference ecosystem for new kernel opportunities, follow [`docs/radar/scouting.md`](docs/radar/scouting.md) — it keeps the [opportunity radar](docs/radar/watchlist.md) current, which feeds the two stages below. To profile a whole model and decide which ops are worth a kernel, follow [`skills/profile-model`](skills/profile-model/SKILL.md) — its output feeds the backlog. To implement a kernel from the [`.issues/kernel/`](.issues/kernel/list.md) backlog, follow [`skills/implement-kernel`](skills/implement-kernel/SKILL.md). The deep guides live in [`docs/guide/`](docs/guide/).

**Local-only, git-ignored:** [`.issues/`](.issues/) (the backlog, including [`.issues/kernel/`](.issues/kernel/)) and [`docs/radar/`](docs/radar/) (the opportunity radar) are in `.gitignore` — they are scratch/working state, not committed. They exist on your machine but won't show in `git status` or land in commits; don't expect them in a fresh clone, and don't try to `git add` them. Everything else under `docs/` and `skills/` is tracked.
