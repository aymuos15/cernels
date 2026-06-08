from functools import cache
from pathlib import Path
from typing import Any


@cache
def load(slug: str) -> Any:
    from kernels import get_local_kernel

    return get_local_kernel(Path(__file__).resolve().parents[1] / slug)
