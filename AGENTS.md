# Agent instructions

## Never run benchmarks or kernels on this machine — only on the Spark

All benchmarking, CUDA builds (`load_inline`), and kernel runs happen on the **Spark** (the GB10 / Grace-Blackwell box, `ssh sie271-pc`), never on the local machine. Transfer the repo with `scripts/transfer.sh sie271-pc`, then run over ssh:

```bash
ssh sie271-pc 'bash -lc "cd ~/kernels && uv run --no-sync python -m benchmark.main <name>"'
```

Do not run `benchmark.main` / `benchmark.view` or compile a `.cu` locally. Pull results back for viewing with `rsync sie271-pc:kernels/analysis/ analysis/`.

Hub kernels (`lib`) have no GB10 / sm_121 build, so on the Spark the `lib` workload is **skipped** (shows `·`) — that's expected; eager / compile / custom still run and verify.

## Workflow

To implement a kernel from the [`.issues/`](.issues/list.md) backlog, follow [`skills/implement-kernel`](skills/implement-kernel/SKILL.md); the deep guides live in [`docs/guide/`](docs/guide/).
