"""Startup setup before huggingface_hub / transformers are imported.

Run via `python -m benchmark.main` so this runs first. It:
1. loads HF_TOKEN from secrets.env (so requests are authenticated when online),
2. defaults to offline so cached benchmarks don't hit the Hub (override with HF_HUB_OFFLINE=0), and
3. bridges a transformers<->kernels version conflict (see below).
"""

import importlib
import os
from pathlib import Path

_secrets = Path("secrets.env")
if _secrets.exists():
    for _line in _secrets.read_text().splitlines():
        _k, _sep, _v = _line.partition("=")
        if _sep and _k.strip() == "HF_TOKEN" and _v.strip():
            os.environ.setdefault("HF_TOKEN", _v.strip())

os.environ.setdefault("HF_HUB_OFFLINE", "1")

# transformers (for native baselines like apply_rotary_pos_emb) pins kernels<0.13, where
# LayerRepository() needs no version; our harness needs kernels>=0.15, where it's required —
# so transformers' import-time kernel mapping raises ValueError. No single kernels version
# satisfies both. Hiding LayerRepository makes transformers' `from kernels import ...` raise
# ImportError, triggering its own graceful "kernels unavailable" fallback. We don't use that
# integration; get_kernel and the benchmark harness are unaffected. (import_module, not an
# `import` statement, so it runs after HF_HUB_OFFLINE is set above.)
_kernels = importlib.import_module("kernels")
if hasattr(_kernels, "LayerRepository"):
    delattr(_kernels, "LayerRepository")
