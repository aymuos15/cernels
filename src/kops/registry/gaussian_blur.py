"""Custom Gaussian blur: a hand-written separable CUDA kernel (gaussian_blur.cu).

Two reflect-101 passes (horizontal then vertical) over (B, C, H, W). The small 1D Gaussian taps
are built per call (matching kornia's formula). The CUDA call is a torch.library custom op
(`kops::gblur`) with a fake impl so torch.compile graphs through it; the thin kernel() wrapper
builds the taps (input prep) then calls it. JIT-built with load_inline on first call.
"""

from importlib.resources import files
from typing import Any

import torch
from torch import Tensor
from torch.utils.cpp_extension import load_inline

_DECL = "at::Tensor gblur(at::Tensor x, at::Tensor ky, at::Tensor kx);"
_SRC = files("kops.registry").joinpath("gaussian_blur.cu").read_text()
_mod: Any = None


def _module():
    global _mod
    if _mod is None:  # compile on first use, not at import
        _mod = load_inline(name="gblur_cuda", cpp_sources=_DECL, cuda_sources=_SRC, functions=["gblur"])
    return _mod


@torch.library.custom_op("kops::gblur", mutates_args=())
def _gblur(x: Tensor, ky: Tensor, kx: Tensor) -> Tensor:
    return _module().gblur(x.contiguous(), ky, kx)


@_gblur.register_fake
def _(x: Tensor, ky: Tensor, kx: Tensor) -> Tensor:
    return torch.empty_like(x)


def _k1d(ks, sigma, device):
    # kornia's gaussian for odd window: x = arange - ks//2, normalized to sum 1.
    x = torch.arange(ks, device=device, dtype=torch.float32) - ks // 2
    g = torch.exp(-(x**2) / (2.0 * sigma**2))
    return g / g.sum()


def kernel(x, kernel_size, sigma):
    kh, kw = kernel_size
    sy, sx = sigma
    return _gblur(x, _k1d(kh, sy, x.device), _k1d(kw, sx, x.device))
