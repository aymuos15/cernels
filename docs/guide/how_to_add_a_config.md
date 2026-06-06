# How to add a config

Create `src/configs/registry/<name>.py` with a `Config` (or `HubConfig`) subclass, then add it to the `CONFIGS` tuple in `registry/__init__.py`. Use `HubConfig` when the op to beat (`lib`) is a Hugging Face Hub kernel; use plain `Config` when it is any other library op (e.g. torchvision) — see [setting up baselines](setting_up_baselines.md).

```python
import torch
from configs.base import HubConfig

class Relu(HubConfig):
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

Because `inputs` and `baseline` are plain methods, anything goes — structured inputs (RoPE's duplicated-half `cos`/`sin`) or composite/lazily-imported baselines (transformers' `apply_rotary_pos_emb`) are just code. See [`registry/rotary.py`](../../src/configs/registry/rotary.py).
