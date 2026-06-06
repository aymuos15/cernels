"""Custom RoPE kernel — starting point (hand-fused + torch.compile).

Swap the body for Triton/CUDA to actually beat the Hub kernel. A custom kernel
module just needs to expose `kernel(*inputs)` matching the config's inputs.
"""

import torch


@torch.compile
def _rope(x, cos, sin):
    d = x.shape[-1] // 2
    rot = torch.cat((-x[..., d:], x[..., :d]), dim=-1)
    return x * cos + rot * sin


def kernel(q, k, cos, sin):
    cos, sin = cos.unsqueeze(1), sin.unsqueeze(1)
    return _rope(q, cos, sin), _rope(k, cos, sin)
