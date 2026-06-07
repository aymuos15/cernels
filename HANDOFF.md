# kernel-builder migration — DONE & committed; remaining = Hub publish

_All four phases are complete, verified on the GB10 Spark, and committed. The only thing left is the actual Hub push (see "Hub push-readiness" below). Nothing has been pushed to the Hub._

## TL;DR

All **9 custom kernels are now HF kernel-builder kernels** (AOT, nix-built, Hub-publishable), replacing the old `load_inline` JIT path. Each is a self-contained repo `src/kops/<name>/` (build.toml + csrc + torch-ext + flake.nix/lock) registering a native `TORCH_LIBRARY` op; a thin `src/kops/registry/<name>.py` loads it via `get_local_kernel`. Built on the GB10 Spark (`scripts/build_kernels.sh`), all 9 **op-bench verify `✓`** and beat/tie `op_compile`. `load_inline` is gone. Docs + skill updated.

## Hub push-readiness (NOT pushed)

Source is complete, builds clean, verifies — push-shaped. Before an actual `nix run .#build-and-upload` (the publish path; uses `[general.hub] repo-id` + `HF_TOKEN`), settle:

1. **`repo-id` is a placeholder** (`kernels-local/<name>` in each `build.toml`). Set the real namespace — the token (`secrets.env`) is user **`aymous`** (orgs: `cai4cai`, `humanitys-last-hackathon`, `brainglobe`). One-line edit per build.toml + decide public/private.
2. **`build/` is aarch64-only** — built on the GB10 Spark, so only the 6 aarch64 variants exist (`torch{211,212}×cu{126,128,130,132}-aarch64-linux`). For non-Spark (x86_64) consumers, build the full matrix (`nix build` / `build-and-upload` produces all supported variants — a much bigger build). aarch64-only is fine if GB10-targeted.
3. **Optional**: a `CARD.md`/README per kernel (Hub model card); currently absent.

`build/` is gitignored + transfer-excluded, so it lives only on the Spark — the push happens from there.

## Final GB10 results (post-migration, all custom ✓)

| kernel | custom vs op_compile | note |
|---|---|---|
| deformable_attention | **34.9×** | also beats Hub kernel; AOT much faster than JIT |
| nms | **2.53×** | AOT build big jump vs JIT (1.4×) |
| roi_align | 1.32× | |
| rmsnorm | 1.23× | |
| rotary | 1.06× | beats compile + Hub kernel |
| primus_3d_rope | 1.07× | |
| gaussian_blur | ~5× vs compile | |
| megablocks_moe | 1.26× | vs the megablocks Hub kernel |
| silu_mul | **0.99× (ties)** | compile fuses this cheap elementwise just as well — honest tie |

(`docs/list.md` has the full normalized table.)

## What changed (all uncommitted — your commit)

- **NEW** `src/kops/<name>/` × 9 (rmsnorm, rope, rope3d, nms, silu_mul, gaussian_blur, roi_align, deform_attn, moe): each `build.toml`, `csrc/<name>.cu`, `torch-ext/torch_binding.{cpp,h}`, `torch-ext/<name>/__init__.py`, `flake.nix`, `flake.lock`.
- **MODIFIED** `src/kops/registry/<name>.py` × 9: now thin `get_local_kernel` loaders (Python prep for nms/gaussian_blur/moe/deform/roi stays here).
- **DELETED** `src/kops/registry/*.cu` × 9 (old load_inline sources; staged via `git rm`).
- **NEW** `scripts/build_kernels.sh` (git-inits each kernel dir + `nix run .#build-and-copy`), `src/kops/.gitignore` (build/, .venv, generated cmake/_ops.py).
- **MODIFIED** `scripts/transfer.sh` (excludes `build/`, `.venv/`, `result`, `torch_compile_debug/` so Spark builds survive `--delete`).
- **MODIFIED** `pyproject.toml` (ruff `extend-exclude` + pyrefly `project-excludes` for kernel-builder torch-ext/build), `docs/list.md`, `docs/guide/how_to_add_a_custom_kernel.md` (rewritten for kernel-builder), `docs/guide/running_benchmarks.md` (build step), `skills/implement-kernel/SKILL.md`.
- `ruff check` and `pyrefly check` on `src/` both pass (the 9 generated `from ._ops import ops` lines carry `# type: ignore`).

## The build/run workflow (Spark-only)

```bash
scripts/transfer.sh sie271-pc                                  # build/ is excluded -> Spark builds persist
ssh sie271-pc 'cd ~/kernels && bash scripts/build_kernels.sh'  # nix build all -> src/kops/<name>/build/
ssh sie271-pc 'cd ~/kernels && uv run --no-sync python -m benchmark.main <name>'
rsync sie271-pc:kernels/analysis/ analysis/ ; uv run --no-sync python -m benchmark.view
```

## Gotchas (all already handled in the committed-to-be files; here so you understand them)

1. flake input is `github:huggingface/kernels` (kernel-builder moved there); entry is `lib.genKernelFlakeOutputs` (was `genFlakeOutputs`).
2. `build.toml [general] name` must be **dashes** (`silu-mul`), not underscores; torch-ext dir + `[kernel.X]` key keep underscores.
3. **Integer op args = `int64_t` in C++** (schema string still `int`). This bit moe (`topk`/`act_id`) and deform_attn (`im2col_step`).
4. `build.toml` needs `version` + `[general.hub] repo-id`, else a misleading `TOML parse error at line 1/2`.
5. The kernel dir must be a **git repo** for nix; `build_kernels.sh` does `git init` per dir. transfer strips `.git` (excluded) so those per-kernel `.git` live only on the Spark — fine, the script re-inits.
6. nix is installed on the Spark, `huggingface.cachix.org` is a system substituter, `ssk23` is trusted → prebuilt torch/cuda download (no source rebuild). The variant we need is `torch212-cxx11-cu130-aarch64-linux`.
7. `build/` is gitignored (push to Hub, never commit) and excluded from transfer (Spark builds persist across syncs).

## Optional follow-ups

- **layers.py + register_fake** per kernel (the kernelize/compile-into-a-model face) — deferred because the whole-model direction is shelved (see below). Add when revisiting models; pattern is `silu-and-mul`'s `op.py` (`add_op_namespace_prefix` + `@custom_op` + `register_fake`).

## Context (shelved, don't resurrect here)

Earlier this session: an end-to-end model investigation (LFM2.5-VL-450M, Qwen3-0.6B reference) concluded batch-1 decode is memory-bandwidth-bound — `torch.compile` ≈1.3× is the practical ceiling, cudagraphs add ~5%, custom kernels win on **compute-bound** ops not memory-bound decode. The `modelkernels` whole-model harness was removed/shelved. This migration is purely about making the existing op-kernels Hub-compliant.
