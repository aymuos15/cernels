"""End-to-end model profiler — find where a whole model spends its time, across archs.

Reuses benchmark's startup bridge (HF_TOKEN, offline default, transformers<->kernels
shim) by importing it: profiling always runs as `python -m profiling.main`, so this
package __init__ runs before transformers is imported.
"""

import os

# Enable Inductor's generated-code dump (output_code.py) for the inductor lens. Must be set
# before torch is imported, so it lives here in the package __init__. Only writes artifacts
# when something is actually torch.compiled (the inductor lens); free otherwise.
os.environ.setdefault("TORCH_COMPILE_DEBUG", "1")

import benchmark  # noqa: E402, F401  (applies the startup bridge before transformers loads)
