# How to add a custom kernel

A custom kernel is your own implementation of an op, benchmarked head-to-head against the reference (op_eager / op_compile) and any Hub kernel (`hub`), and verified against the config's reference. It lives in `src/kops/` and is wired in through a config.

## 1. Write the kernel

Create `src/kops/registry/<name>.py` exposing `kernel(*inputs)` — taking the same inputs the config builds (what the Hub `op` receives) and returning the result (a tensor, or a tuple):

```python
# src/kops/registry/<name>.py
def kernel(q, k, cos, sin):
    ...
    return q_out, k_out
```

Pure torch or a hand-written CUDA kernel — anything that runs on the inputs.

## 2. Wire it into a config

In the kernel's `Config` subclass (`src/configs/registry/<name>.py`), set `custom`:

```python
from kops.registry.rope import kernel as rope_kernel

class Rotary(HubConfig):
    ...
    custom = staticmethod(rope_kernel)
```

Use `staticmethod` so it's called as `kernel(*inputs)`, not bound to the config. The runner then adds a `custom` workload alongside `eager` / `compile` / `lib`, checked against `baseline` the same way the `lib` op is.

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
- The `.cu` is auto-formatted by the clang-format pre-commit hook.
