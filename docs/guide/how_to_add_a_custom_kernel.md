# How to add a custom kernel

A custom kernel is your own implementation of an op, benchmarked head-to-head against the Hub kernel and torch (eager / compile) and verified against the config's `baseline`. It lives in `src/kops/` and is wired in through a config.

## 1. Write the kernel

Create `src/kops/<name>.py` exposing `kernel(*inputs)` — taking the same inputs the config builds (what the Hub `op` receives) and returning the result (a tensor, or a tuple):

```python
# src/kops/<name>.py
def kernel(q, k, cos, sin):
    ...
    return q_out, k_out
```

Pure torch or a hand-written CUDA kernel — anything that runs on the inputs.

## 2. Wire it into a config

In the kernel's `Config` subclass (`src/configs/registry/<name>.py`), set `custom`:

```python
from kops.rope import kernel as rope_kernel

class Rotary(Config):
    ...
    custom = staticmethod(rope_kernel)
```

Use `staticmethod` so it's called as `kernel(*inputs)`, not bound to the config. The runner then adds a `custom` workload alongside `eager` / `compile` / `kernel`, checked against `baseline` the same way the Hub kernel is.

## 3. Run

```bash
uv run --no-sync python -m benchmark.main <name>
```

The `custom` column shows up in the results table (and in `python -m benchmark.view`).

## CUDA example

[`rope.cu`](../../src/kops/rope.cu) + [`rope.py`](../../src/kops/rope.py) are a worked example: a fused CUDA RoPE kernel JIT-compiled with torch's `load_inline`. `rope.py` reads `rope.cu` and builds it on first call (cached afterwards, so other benchmarks stay fast). On an RTX A1000 it runs ~19 ms vs the Hub kernel's ~35 ms and torch.compile's ~23 ms.
