---
name: profile-model
description: Operating manual to profile a whole model end-to-end (prefill + decode, three lenses) and produce a ranked shortlist of canonical ops worth a custom kernel. Launch with a model name; everything else is here.
---

# Profile a model (decide what to optimize)

Take one model to a saved profile + a ranked recommendation of which **canonical ops** (RMSNorm, RoPE, CausalConv1d, SiluAndMul, …) are worth a custom kernel. This is the front of the funnel: its output feeds [`implement-kernel`](../implement-kernel/SKILL.md). **Read this whole file and the linked guides before running — don't guess.**

> **Never run profilers or build kernels locally — only on the Spark** (`ssh spark`). See [AGENTS.md](../../AGENTS.md).

## Launching (for whoever starts the agent)
The invocation is a model name plus this skill: **`@skills/profile-model/SKILL.md profile <model>`** — `<model>` is a `ModelProfile.name` in [`src/profiling/registry/`](../../src/profiling/registry/). If the model isn't there yet, adding it is step 1.

## Rules
- **Spark-only** for any run: edit locally → `scripts/transfer.sh spark` → run over ssh → `rsync spark:kernels/analysis/ analysis/` to pull results back. The Spark uses a password — if auth hangs, note it and stop; don't retry forever.
- **Don't `git commit`/`push`.** Leave changes in the working tree for review.
- **No monkeypatching.** Profiling only observes the eager model (forward hooks + `torch.profiler` + an Inductor dump); it must not alter forwards. Kernel integration is a separate workflow.
- **Profile, don't micro-benchmark.** One representative pass per phase is enough; the goal is *where the time goes*, not a precise speedup number.

## Conventions
- **Model name** = the `ModelProfile.name` (e.g. `lfm2_5_vl_450m`); it's the CLI arg to `profiling.main`.
- **ModelProfile** → `src/profiling/registry/<name>.py`, added to `MODELS` in `src/profiling/registry/__init__.py`. It defines `load()` (model + processor) and `inputs()` (one representative batch; for a VLM, an image + prompt). Study [`registry/lfm2_5_vl_450m.py`](../../src/profiling/registry/lfm2_5_vl_450m.py) and [`registry/base.py`](../../src/profiling/registry/base.py).
- **Output** → `analysis/<host>/profile/<name>.{json,txt}` + dumped Triton under `.../profile/inductor/`.

## Steps

### 0. Add or confirm the ModelProfile
If `<model>` is already in `MODELS`, read its file. If not, write `src/profiling/registry/<name>.py`: a `ModelProfile` subclass with `load()` and `inputs()`. Match the loader to the model card (`AutoModelForImageTextToText` + `AutoProcessor` for a VLM, `AutoModelForCausalLM` + `AutoTokenizer` for an LLM); set `decode_tokens`. The package reuses the `benchmark` startup bridge, so transformers loads despite the kernels version conflict — see [`src/profiling/__init__.py`](../../src/profiling/__init__.py).

### 1. Run it on the Spark
`scripts/transfer.sh spark`, then:
```
ssh spark 'bash -lc "cd ~/kernels && HF_HUB_OFFLINE=0 uv run --no-sync python -m profiling.main <model>"'
rsync spark:kernels/analysis/ analysis/
uv run --no-sync python -m profiling.view <model>      # render locally (read-only)
```
`HF_HUB_OFFLINE=0` is needed the first time to download the model. Add `--no-inductor` to skip the compile/dump lens for a fast first pass. See [`src/profiling/README.md`](../../src/profiling/README.md) and [profiling on the Spark](../../docs/guide/profiling_on_spark.md) (GB10/UMA quirks: lead with timeline + tok/s, don't over-trust VRAM, `ncu` counter perms).

### 2. Read the three lenses (correctly)
Per phase (**prefill** = first forward, includes vision encode for a VLM; **decode** = per-token steps):
- **ops** (`torch.profiler`) — the **authoritative GPU-time** breakdown. Trust this for "which kernel dominates."
- **modules** — self time per layer class, maps to **canonical kernelize names**. Caveat: large self% on container/`Embedding` classes means **CPU launch-gap (launch-bound decode)**, not compute — don't mistake it for a kernel opportunity. When this dominates, the real lever is CUDA graphs / `torch.compile(mode="reduce-overhead")`, not a hand kernel.
- **inductor** — the generated Triton (under `.../profile/inductor/`) shows **what `torch.compile` already fuses**. Read it before proposing a kernel: don't hand-write what Inductor already nails.

### 3. Decide what's worth a kernel
For each significant op, cross-reference the lenses and classify:
- **write a kernel** — meaningful GPU-time share *and* Inductor doesn't already fuse it well *and* it's not purely launch-bound.
- **already fused — skip** — Inductor's Triton already covers it.
- **not the bottleneck** — small share at the phase that matters (decode usually).
Map each to a **canonical op name** (the name a kernel would register under), not the model-specific class.

## Report (definition of done)
Done when the profile runs on the Spark and is saved under `analysis/<host>/profile/`. Work quietly; return a **concise final report** with exactly:
1. **Profile location** + model/machine.
2. **Phase breakdown** — for prefill and decode, the top contributors from the ops lens (authoritative) with %, and the module-lens mapping to canonical ops. Call out explicitly if a phase is **launch-bound**.
3. **Ranked shortlist** — canonical ops worth a kernel, each with evidence (self% + phase) and a verdict: *write a kernel* / *already-fused-skip* / *not-the-bottleneck*.
4. **Inductor note** — what compile already fuses (so we don't duplicate it).
5. **Hand-off** — the candidate ops, phrased so [`implement-kernel`](../implement-kernel/SKILL.md) can pick them up.
