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

Then point a config at it with `custom: <name>` (see [`src/configs/`](../configs/README.md)). Start from torch, then swap in Triton/CUDA to actually beat the Hub kernel.
