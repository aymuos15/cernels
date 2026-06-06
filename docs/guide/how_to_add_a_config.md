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

## Non-Hub config (the common backlog case)

Most backlog issues have no Hub kernel — the baseline *is* the target and a `custom` kernel is the contender. Subclass plain `Config` (no `repo`, `lib` stays unset) and override `verify` if the output isn't a single comparable tensor:

```python
import torch, torchvision
from configs.base import Config
from kops.registry.nms import kernel as nms_kernel

class NMS(Config):
    name = "nms"
    dtype = torch.float32
    op = "torchvision.ops.nms"        # label only (no Hub kernel)
    use_compile = False               # data-dependent output -> compile only graph-breaks
    custom = staticmethod(nms_kernel)

    def inputs(self, device, dtype):
        ...                           # boxes, scores, iou_threshold
        return boxes, scores, 0.5

    def baseline(self, boxes, scores, iou):
        return torchvision.ops.nms(boxes, scores, iou)

    def verify(self, out, ref):       # index sets, not allclose
        return set(out.tolist()) == set(ref.tolist())
```

See [`registry/nms.py`](../../src/configs/registry/nms.py), [setting up baselines](setting_up_baselines.md), and [correctness](correctness.md).

## Fields

| field | who | meaning |
|---|---|---|
| `name` | both | registry key / CLI arg |
| `dtype` | both | input dtype (default `float16`) |
| `op` | both | label for the op (shown in saved records); on `HubConfig` it's the attribute called on the loaded kernel |
| `use_compile` | both | include the `torch.compile` workload (default `True`; set `False` for data-dependent ops) |
| `inputs(device, dtype)` | both | returns the input tuple (must also work on the `meta` device for shape capture) |
| `baseline(*inputs)` | both | native eager reference + correctness reference |
| `verify(out, ref)` | both | correctness check (default `allclose` atol 1e-2); override for non-tensor outputs |
| `custom` | both | optional `staticmethod(fn)` benchmarked as the `custom` workload |
| `lib` | `Config` | the production op to beat; leave unset when `baseline` already is it |
| `repo` / `version` | `HubConfig` | Hub kernel repo id / revision (version defaults to 1) |
| `out_arg` | `HubConfig` | kernel writes into a leading `out` tensor (`True`) vs returns the result (`False`) |
