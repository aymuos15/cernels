# kops

Custom kernels (kernel-ops) — your own implementations, benchmarked head-to-head against the reference (op_eager / op_compile) and any Hub kernel (`hub`).

Named `kops` (not `kernels`) on purpose: a top-level `kernels` package would shadow the Hugging Face `kernels` library the project imports.

Each kernel is a self-contained [kernel-builder](https://github.com/huggingface/kernel-builder) repo `src/kops/<slug>/` (build.toml, csrc/, torch-ext/, flake.{nix,lock}, CARD.md), built AOT via nix on the Spark, plus a thin loader in [`registry/`](registry/) (`src/kops/registry/<slug>.py`) that a config wires as its `custom` contender. `src/kops/rms_norm/` + `registry/rms_norm.py` are a minimal worked example.

Naming follows [RULES.md §1](../../RULES.md) (one canonical slug per kernel, enforced by `scripts/check_naming.py`). To add one, see [docs/guide/how_to_add_a_custom_kernel.md](../../docs/guide/how_to_add_a_custom_kernel.md).
