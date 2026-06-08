"""Base config classes; each kernel/op is a Config subclass in configs/registry/.

Workloads (see docs/guide/setting_up_baselines.md): only the reference gets the
op_eager/op_compile split (the sole workload torch.compile can fuse); hub/lib/custom are
each timed once. With `reference_is_hub`, `baseline` is timed as `hub` and there is no
op_eager/op_compile.
"""

from collections.abc import Callable
from typing import Any

import torch

from kernels import get_kernel


def _default_kernel_layer_version() -> None:
    # kernels 0.15 rejects LayerRepository/FuncRepository constructed without a version;
    # transformers' Hub-kernel modules construct them without one when imported. Default to
    # version=1 so any transformers-referenced config imports. Idempotent; not in the timed path.
    import kernels.layer.func as _kf
    import kernels.layer.layer as _kl

    for cls in (_kl.LayerRepository, _kf.FuncRepository):
        if getattr(cls, "_kops_version_defaulted", False):
            continue
        orig = cls.__init__

        def make(orig):
            def patched(self, *a, **k):
                if k.get("revision") is None and k.get("version") is None:
                    k["version"] = 1
                return orig(self, *a, **k)

            return patched

        cls.__init__ = make(orig)
        cls._kops_version_defaulted = True


class Config:
    def __init__(self) -> None:
        _default_kernel_layer_version()

    name: str = ""
    dtype: torch.dtype = torch.float16
    op: str = ""
    use_compile: bool = True  # off for data-dependent ops torch.compile can't trace
    reference_is_hub: bool = False  # baseline IS a Hub kernel -> time it as `hub`, no op_eager/op_compile
    hub: Any = None
    lib: Any = None
    custom: Any = None
    inputs: Callable[..., tuple]  # (device, dtype) -> input tuple
    baseline: Callable[..., Any]

    def verify(self, out, ref) -> bool:
        """Override for non-tensor outputs."""
        return bool(torch.allclose(out, ref, atol=1e-2))


class HubConfig(Config):
    """`hub` contender is a Hub kernel: get_kernel(repo, version).<op>(...)."""

    repo: str = ""
    version: int = 1
    out_arg: bool = False  # kernel writes into a leading out tensor vs returns the result
    kernel: Any = None

    def hub(self, *inputs):
        if self.kernel is None:
            self.kernel = get_kernel(self.repo, version=self.version)
        fn = getattr(self.kernel, self.op)
        if self.out_arg:
            out = torch.empty_like(inputs[0])
            fn(out, *inputs)
            return out
        return fn(*inputs)
