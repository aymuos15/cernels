"""Whole-model integration: swap kops kernels into a transformers model and benchmark end-to-end.

Reuses benchmark's startup bridge (HF_TOKEN, offline default, transformers<->kernels shim):
modeling always runs as `python -m modeling.main`, so this __init__ runs before transformers.
"""

import benchmark  # noqa: F401  (applies the startup bridge before transformers loads)
