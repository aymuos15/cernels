# How to add a config

Create `src/configs/registry/<name>.py` with a `Config` (or `HubConfig`) subclass, then add it to the `CONFIGS` tuple in `registry/__init__.py`. Use `HubConfig` when there is a Hugging Face Hub kernel contender (`hub`); use plain `Config` otherwise — see [setting up baselines](setting_up_baselines.md) for the full `op_eager`/`op_compile`/`hub`/`lib`/`custom` model.

```python
import torch
from configs.base import HubConfig

class Relu(HubConfig):
    name = "relu"
    repo = "kernels-community/relu"   # Hub kernel repo (version defaults to 1)
    dtype = torch.float32
    op = "relu"                       # attribute on the loaded kernel (the `hub` contender)
    out_arg = True                    # k.op(out, *inputs); False -> result = k.op(*inputs)

    def inputs(self, device, dtype):  # the op's input tensors, any structure
        return (torch.randn(4096, 4096, device=device, dtype=dtype),)

    def baseline(self, x):            # the reference op (op_eager) + correctness reference
        return torch.relu(x)
```

`HubConfig` supplies the `hub` contender automatically (it loads `repo` and calls the `op` attribute). Because `inputs` and `baseline` are plain methods, anything goes — structured inputs (RoPE's duplicated-half `cos`/`sin`) or composite/lazily-imported references (transformers' `apply_rotary_pos_emb`) are just code. See [`registry/rotary_embedding.py`](../../src/configs/registry/rotary_embedding.py).

## Non-Hub config (the common backlog case)

Most backlog issues have no Hub kernel — the reference op *is* the target and a `custom` kernel is the contender. Subclass plain `Config` (no `repo`, `hub`/`lib` stay unset) and override `verify` if the output isn't a single comparable tensor:

```python
import torch, torchvision
from configs.base import Config
from kops.registry.non_maximum_suppression import kernel as nms_kernel

class NonMaximumSuppression(Config):     # CamelCase of the slug (RULES.md §1)
    name = "non_maximum_suppression"     # canonical slug; the build/dir/op/repo all derive from it
    dtype = torch.float32
    op = "torchvision.ops.nms"        # label only (no Hub kernel)
    use_compile = False               # data-dependent output -> op_compile only graph-breaks
    custom = staticmethod(nms_kernel)

    def inputs(self, device, dtype):
        ...                           # boxes, scores, iou_threshold
        return boxes, scores, 0.5

    def baseline(self, boxes, scores, iou):   # the reference op (op_eager)
        return torchvision.ops.nms(boxes, scores, iou)

    def verify(self, out, ref):       # index sets, not allclose
        return set(out.tolist()) == set(ref.tolist())
```

See [`registry/non_maximum_suppression.py`](../../src/configs/registry/non_maximum_suppression.py), [setting up baselines](setting_up_baselines.md), and [correctness](correctness.md).

## Fields

| field | who | meaning |
|---|---|---|
| `name` | both | registry key / CLI arg |
| `dtype` | both | input dtype (default `float16`) |
| `op` | both | label for the op (shown in saved records); on `HubConfig` it's the attribute called on the loaded kernel for the `hub` contender |
| `use_compile` | both | include the `op_compile` workload (default `True`; set `False` for data-dependent ops) |
| `reference_is_hub` | `Config` | `True` when `baseline` IS a Hub kernel — it's timed as `hub`, with no `op_eager`/`op_compile` (e.g. `megablocks_moe`); default `False` |
| `inputs(device, dtype)` | both | returns the input tuple (must also work on the `meta` device for shape capture) |
| `baseline(*inputs)` | both | the reference op (the `op_eager` workload, or `hub` if `reference_is_hub`) + correctness reference |
| `verify(out, ref)` | both | correctness check (default `allclose` atol 1e-2); override for non-tensor outputs |
| `custom` | both | optional `staticmethod(fn)` benchmarked as the `custom` workload |
| `hub` | `Config` | optional Hub-kernel contender (`hub` workload); `HubConfig` supplies this automatically from `repo`/`op` |
| `lib` | `Config` | optional separate library impl (`lib` workload), distinct from the reference; usually unset |
| `repo` / `version` | `HubConfig` | Hub kernel repo id / revision (version defaults to 1) |
| `out_arg` | `HubConfig` | kernel writes into a leading `out` tensor (`True`) vs returns the result (`False`) |
