# kops

Custom kernels (kernel-ops) — your own implementations, benchmarked head-to-head against the Hub kernel and torch (eager / compile).

Named `kops` (not `kernels`) on purpose: a top-level `kernels` package would shadow the Hugging Face `kernels` library the project imports.

## Adding one

Create `src/kops/<name>.py` exposing `kernel(*inputs)` — same inputs as the Hub `op`, returning the result:

```python
def kernel(q, k, cos, sin):
    ...
    return q_out, k_out
```

Then point a config at it with `custom: <name>` (see [`src/configs/`](../configs/README.md)).

[`rope.py`](rope.py) is a worked example: a fused CUDA RoPE kernel JIT-compiled with torch's `load_inline` (built on first call, cached after). On an RTX A1000 it runs ~19 ms vs the Hub kernel's ~35 ms and torch.compile's ~23 ms.
