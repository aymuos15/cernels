"""Base config classes. Each kernel/op is a Config subclass in configs/registry/.

- Config: the general interface — a benchmark of an op, comparing eager / compile /
  lib (the production reference) / custom (your own). The reference `lib` can be
  anything (a Hub kernel, torchvision, ...), so ops that aren't on the Hub fit too.
- HubConfig: convenience base whose `lib` is a Hugging Face Hub kernel.
"""

from collections.abc import Callable
from typing import Any

import torch

from kernels import get_kernel


class Config:
    name: str = ""  # registry key (CLI arg)
    dtype: torch.dtype = torch.float16
    op: str = ""  # label for the reference op (shown in saved records)
    use_compile: bool = True  # include the torch.compile workload (off for data-dependent ops)
    custom: Any = None  # optional callable -> benchmarked as the `custom` workload
    lib: Any = None  # the production op to beat (the `lib` workload); None when `baseline` already is it
    inputs: Callable[..., tuple]  # (device, dtype) -> input tuple
    baseline: Callable[..., Any]  # native eager reference (the `eager` workload)

    def verify(self, out, ref) -> bool:
        """Correctness of a workload vs the eager baseline. Override for non-tensor outputs."""
        return bool(torch.allclose(out, ref, atol=1e-2))


class HubConfig(Config):
    """`lib` is a Hub kernel: get_kernel(repo, version).<op>(...)."""

    repo: str = ""
    version: int = 1
    out_arg: bool = False  # kernel writes into a leading out tensor vs returns the result
    kernel: Any = None  # the loaded kernel module (set on first lib call)

    def lib(self, *inputs):
        if self.kernel is None:
            self.kernel = get_kernel(self.repo, version=self.version)
        fn = getattr(self.kernel, self.op)
        if self.out_arg:
            out = torch.empty_like(inputs[0])
            fn(out, *inputs)
            return out
        return fn(*inputs)
