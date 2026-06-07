"""Base config classes. Each kernel/op is a Config subclass in configs/registry/.

A benchmark compares up to five workloads (see docs/guide/setting_up_baselines.md):
  - op_eager / op_compile : the reference op (a torch-op composition), run eager and
    torch.compiled. The correctness ground-truth and speedup denominator. Only the
    reference gets the compile split — it's the only workload torch.compile can fuse.
  - hub    : an HF Kernel Hub kernel (kernels-community/...) for the op.
  - lib    : a separate library implementation of the op, distinct from the reference.
  - custom : our own kernel (src/kops/...). Always a contender, never the reference.

- Config: the general interface. `baseline` is the reference; `hub`/`lib`/`custom` are
  optional contenders, each timed once (torch.compile only graph-breaks around them).
- HubConfig: convenience base whose `hub` contender is a Hugging Face Hub kernel.

When no library op exists and a Hub kernel IS the reference (e.g. megablocks_moe), set
`reference_is_hub = True`: `baseline` is then timed as the `hub` workload and there is no
op_eager/op_compile.
"""

from collections.abc import Callable
from typing import Any

import torch

from kernels import get_kernel


class Config:
    name: str = ""  # registry key (CLI arg)
    dtype: torch.dtype = torch.float16
    op: str = ""  # label for the reference op (shown in saved records)
    use_compile: bool = True  # include the op_compile workload (off for data-dependent ops)
    reference_is_hub: bool = False  # baseline IS a Hub kernel -> time it as `hub`, no op_eager/op_compile
    hub: Any = None  # optional HF Hub kernel contender -> the `hub` workload
    lib: Any = None  # optional separate library impl -> the `lib` workload
    custom: Any = None  # optional callable -> the `custom` workload
    inputs: Callable[..., tuple]  # (device, dtype) -> input tuple
    baseline: Callable[..., Any]  # the reference op (op_eager workload, or `hub` if reference_is_hub)

    def verify(self, out, ref) -> bool:
        """Correctness of a contender vs the reference output. Override for non-tensor outputs."""
        return bool(torch.allclose(out, ref, atol=1e-2))


class HubConfig(Config):
    """`hub` contender is a Hub kernel: get_kernel(repo, version).<op>(...)."""

    repo: str = ""
    version: int = 1
    out_arg: bool = False  # kernel writes into a leading out tensor vs returns the result
    kernel: Any = None  # the loaded kernel module (set on first hub call)

    def hub(self, *inputs):
        if self.kernel is None:
            self.kernel = get_kernel(self.repo, version=self.version)
        fn = getattr(self.kernel, self.op)
        if self.out_arg:
            out = torch.empty_like(inputs[0])
            fn(out, *inputs)
            return out
        return fn(*inputs)
