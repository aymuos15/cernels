"""Base config class. Each kernel is a Config subclass in configs/registry/."""

from collections.abc import Callable
from typing import Any

import torch


class Config:
    """Subclasses define two methods (free to use any signature):

    - inputs(self, device, dtype) -> tuple   : the op's input tensors, any structure
    - baseline(self, *inputs)               : native eager reference + correctness check
    """

    name: str = ""  # registry key (CLI arg)
    repo: str = ""  # Hub kernel repo id
    version: int = 1
    dtype: torch.dtype = torch.float16
    op: str = ""  # attribute on the loaded kernel to call
    out_arg: bool = False  # kernel writes into a leading `out` tensor vs returns the result
    custom: Any = None  # optional staticmethod(callable) -> benchmarked as an extra contestant
    inputs: Callable[..., tuple]
    baseline: Callable[..., Any]

    def out_buffer(self, inputs):
        """Preallocated output for `out_arg` kernels. Override when it isn't shaped
        like inputs[0] (e.g. gated activations output half the last dim)."""
        return torch.empty_like(inputs[0])
