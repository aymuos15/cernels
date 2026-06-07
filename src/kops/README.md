# kops

Custom kernels (kernel-ops) — your own implementations, benchmarked head-to-head against the reference (op_eager / op_compile) and any Hub kernel (`hub`).

Named `kops` (not `kernels`) on purpose: a top-level `kernels` package would shadow the Hugging Face `kernels` library the project imports.

Kernels live in [`registry/`](registry/). [`rope.cu`](registry/rope.cu) + [`rope.py`](registry/rope.py) are a worked example: a fused CUDA RoPE kernel JIT-compiled with torch's `load_inline`.

To add one, see [docs/guide/how_to_add_a_custom_kernel.md](../../docs/guide/how_to_add_a_custom_kernel.md).
