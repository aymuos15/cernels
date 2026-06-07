"""gaussian_blur loader — kernel-builder kernel under src/kops/gaussian_blur/ (built with nix; see scripts/build_kernels.sh)."""

from pathlib import Path
from typing import Any

import torch

_mod: Any = None
_REPO = Path(__file__).resolve().parents[1] / "gaussian_blur"


def _module():
    global _mod
    if _mod is None:
        from kernels import get_local_kernel

        _mod = get_local_kernel(_REPO)
    return _mod


def _k1d(ks, sigma, device):
    x = torch.arange(ks, device=device, dtype=torch.float32) - ks // 2
    g = torch.exp(-(x**2) / (2.0 * sigma**2))
    return g / g.sum()


def kernel(x, kernel_size, sigma):
    kh, kw = kernel_size
    sy, sx = sigma
    return _module().gblur(x.contiguous(), _k1d(kh, sy, x.device), _k1d(kw, sx, x.device))
