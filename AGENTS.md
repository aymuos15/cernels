# Agent instructions

## Never run benchmarks or kernels on this machine — only on the Spark

All benchmarking, CUDA builds (`load_inline`), and kernel runs happen on the **Spark** (the GB10 / Grace-Blackwell box, `ssh sie271-pc`), never on the local machine. Transfer the repo with `scripts/transfer.sh sie271-pc`, then run over ssh:

```bash
ssh sie271-pc 'bash -lc "cd ~/kernels && uv run --no-sync python -m benchmark.main <name>"'
```

Do not run `benchmark.main` / `benchmark.view` or compile a `.cu` locally. Pull results back for viewing with `rsync sie271-pc:kernels/analysis/ analysis/`.

Hub kernels (the `hub` workload) generally **do** build on the Spark — current `kernels-community` repos ship a Version 1 `...cu130-aarch64-linux` build that resolves on GB10 / sm_121. Only when no compatible build exists is the `hub` workload **skipped** (shows `·`); op_eager / op_compile / custom still run and verify. See [docs/guide/running_benchmarks.md](docs/guide/running_benchmarks.md).

## Workflow

To implement a kernel from the [`.issues/`](.issues/list.md) backlog, follow [`skills/implement-kernel`](skills/implement-kernel/SKILL.md); the deep guides live in [`docs/guide/`](docs/guide/).
