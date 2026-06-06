# configs

One `Config` subclass per kernel, living in [`registry/`](registry/) and listed in
[`registry/__init__.py`](registry/__init__.py).

```bash
uv run --no-sync python -m benchmark.main <name>   # <name> = a Config.name
```

## Adding a kernel

Create `registry/<name>.py` with a `Config` subclass, then add it to the `CONFIGS`
tuple in `registry/__init__.py`.

```python
import torch
from configs.base import Config

class Relu(Config):
    name = "relu"
    repo = "kernels-community/relu"   # Hub kernel repo (version defaults to 1)
    dtype = torch.float32
    op = "relu"                       # attribute on the loaded kernel
    out_arg = True                    # k.op(out, *inputs); False -> result = k.op(*inputs)

    def inputs(self, device, dtype):  # the op's input tensors, any structure
        return (torch.randn(4096, 4096, device=device, dtype=dtype),)

    def baseline(self, x):            # native eager reference + correctness check
        return torch.relu(x)
```

Because `inputs` and `baseline` are plain methods, anything goes — structured inputs
(RoPE's duplicated-half `cos`/`sin`) or composite/lazily-imported baselines (transformers'
`apply_rotary_pos_emb`) are just code. See [`registry/rotary.py`](registry/rotary.py).

## custom kernels

Set `custom = staticmethod(fn)` to benchmark your own kernel (from [`src/kops/`](../kops/README.md))
as an extra `custom` workload alongside `eager` / `compile` / `kernel`, verified against
`baseline`. This is how you try to *beat* a Hub kernel.
