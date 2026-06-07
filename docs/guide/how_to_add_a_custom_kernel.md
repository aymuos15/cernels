# How to add a custom kernel

A custom kernel is your own implementation of an op, benchmarked head-to-head against the reference (op_eager / op_compile) and any Hub kernel (`hub`), and verified against the config's reference. It lives in `src/kops/` and is wired in through a config.

## 1. Write the kernel — as a `torch.library` custom op (required)

Create `src/kops/registry/<name>.py` exposing `kernel(*inputs)` registered as a **`torch.library` custom op with a `register_fake`**. This is mandatory, not optional: it's what lets the kernel be dropped into a real model and survive `torch.compile` (compile graphs *through* a registered op with no graph break; an opaque `load_inline` function forces a graph break and is *worse* than no kernel — see [modelkernels](../../src/modelkernels/README.md)). The fake (meta) impl returns output tensors of the right shape/dtype without running, so compile can trace shapes.

```python
# src/kops/registry/<name>.py
import torch
from torch import Tensor

@torch.library.custom_op("kops::myop", mutates_args=())
def kernel(q: Tensor, k: Tensor, cos: Tensor, sin: Tensor) -> tuple[Tensor, Tensor]:
    return _module().myop(q, k, cos, sin)   # the CUDA call (load_inline, lazily built)

@kernel.register_fake
def _(q: Tensor, k: Tensor, cos: Tensor, sin: Tensor) -> tuple[Tensor, Tensor]:
    return torch.empty_like(q), torch.empty_like(k)
```

Typed signature is required (custom_op infers the schema from annotations; args may be `Tensor`/`int`/`float`/`bool`). **Keep any Python prep (argsort, building taps, routing) in a thin `kernel()` wrapper and register only the CUDA call as the custom op** — see [`nms.py`](../../src/kops/registry/nms.py) (data-dependent fake via `get_ctx().new_dynamic_size()`), [`gaussian_blur.py`](../../src/kops/registry/gaussian_blur.py) / [`moe.py`](../../src/kops/registry/moe.py) (prep + op split). For a data-independent output, the fake is usually just `torch.empty_like` / `value.new_empty((...))`.

## 2. Wire it into a config

In the kernel's `Config` subclass (`src/configs/registry/<name>.py`), set `custom`:

```python
from kops.registry.rope import kernel as rope_kernel

class Rotary(HubConfig):
    ...
    custom = staticmethod(rope_kernel)
```

Use `staticmethod` so it's called as `kernel(*inputs)`, not bound to the config. The runner then adds a `custom` workload alongside `op_eager` / `op_compile` / `hub` / `lib`, checked against the reference the same way the `hub`/`lib` ops are.

## 3. Run

```bash
uv run --no-sync python -m benchmark.main <name>
```

The `custom` column shows up in the results table (and in `python -m benchmark.view`).

## CUDA example

[`rope.cu`](../../src/kops/registry/rope.cu) + [`rope.py`](../../src/kops/registry/rope.py) are a worked example: a fused CUDA RoPE kernel JIT-compiled with torch's `load_inline`. `rope.py` reads `rope.cu` and builds it on first call (cached afterwards, so other benchmarks stay fast). On an RTX A1000 it runs ~19 ms vs the Hub kernel's ~35 ms and torch.compile's ~23 ms.

## Gotchas

- **Run via `uv run`.** `load_inline` needs `ninja` on `PATH`; `uv run` puts `.venv/bin` there, running `.venv/bin/python` directly does not.
- **Read the `.cu` with `files("kops.registry")`** (not `files("kops")`) — the `.py`/`.cu` pair lives in `src/kops/registry/`. Mirror `rope.py`'s build/binding glue.
- **Reduce in fp32** inside the kernel even for fp16/bf16 inputs, or `verify` will fail on tolerance.
- **Return on the right device.** Allocate outputs with `at::TensorOptions().device(...)` that matches what the caller expects — a host `memcpy` into a CUDA tensor segfaults.
- **`register_fake` must match the real output** shape/dtype, or compile will mis-trace. Don't read tensor data in the fake (it runs on meta tensors).
- **`mutates_args=()`** — kernels must be functional (allocate fresh outputs, don't mutate inputs); aliasing an input breaks the custom-op contract.
- The `torch.library` wrapper adds a few-µs eager dispatch cost (negligible except on sub-100µs kernels, and gone under compile) — worth it for compile-composability.
- The `.cu` is auto-formatted by the clang-format pre-commit hook.
