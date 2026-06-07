"""ModelProfile: the cross-arch unit of the profiler (the analog of configs.Config).

A ModelProfile knows how to load a model + processor and build one representative
input batch. The engine drives the two phases (prefill, decode) from there. Adding a
new architecture = one small subclass in registry/, exactly like adding a kernel config.
"""

from typing import Any

import torch


class ModelProfile:
    name: str = ""  # registry key (CLI arg)
    model_id: str = ""  # HF repo id
    dtype: torch.dtype = torch.bfloat16
    decode_tokens: int = 32  # number of decode steps to profile
    compile_submodule: str = ""  # dotted attr path to compile for the inductor lens ("" = whole model)

    def load(self, device) -> tuple[Any, Any]:
        """Return (model, processor_or_tokenizer). Override per arch."""
        raise NotImplementedError

    def inputs(self, processor, device) -> dict:
        """Return the kwargs dict for a prefill forward (input_ids, pixel_values, ...)."""
        raise NotImplementedError

    def submodule(self, model):
        """Resolve compile_submodule to an actual module (or the model itself)."""
        mod = model
        for part in self.compile_submodule.split(".") if self.compile_submodule else []:
            mod = getattr(mod, part)
        return mod
