import torch

from configs.base import Config
from kops.registry.rope import kernel as rope_kernel


class Rotary(Config):
    name = "rotary"
    repo = "kernels-community/rotary"
    dtype = torch.float16
    op = "apply_rotary_transformers"
    custom = staticmethod(rope_kernel)

    def inputs(self, device, dtype):
        # RoPE requires cos/sin with duplicated halves (cos[:d/2] == cos[d/2:]); not plain randn.
        b, h, s, d = 2, 32, 2048, 128
        q = torch.randn(b, h, s, d, device=device, dtype=dtype)
        k = torch.randn(b, h, s, d, device=device, dtype=dtype)
        ang = torch.randn(b, s, d // 2, device=device, dtype=dtype)
        cos = torch.cat([ang.cos(), ang.cos()], dim=-1)
        sin = torch.cat([ang.sin(), ang.sin()], dim=-1)
        return q, k, cos, sin

    def baseline(self, q, k, cos, sin):
        # transformers' canonical RoPE (imported lazily; see benchmark/__init__ for the kernels bridge)
        from transformers.models.llama.modeling_llama import apply_rotary_pos_emb

        return apply_rotary_pos_emb(q, k, cos, sin)
