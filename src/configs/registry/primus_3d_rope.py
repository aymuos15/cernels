"""Primus 3D axial RoPE benchmark config.

Baseline is the exact timm path Primus uses (`RotaryEmbeddingCat` + `apply_rot_embed_cat`),
no hand-written rope. The Primus package `dynamic-network-architectures` pins `timm<1.0.23`,
the version where 3D `RotaryEmbeddingCat` works — newer timm (>=1.0.23) has a 3D
buffer-shape bug.

Timed path: the rotation of q and k tensors using the precomputed embed.
Precomputed in inputs(): the rope embed (sin/cos cat), which is legitimately cached once per
fixed grid in the real model and shared across all blocks (exactly as `Eva._pos_embed` calls
`self.rope.get_embed()` once per forward).
"""

import torch

from configs.base import Config
from kops.registry.primus_3d_rope import kernel as rope3d_kernel


class Primus3dRope(Config):
    name = "primus_3d_rope"
    dtype = torch.bfloat16
    op = "timm.layers.apply_rot_embed_cat"
    custom = staticmethod(rope3d_kernel)

    # Primus-M: embed_dim=792, n_heads=12, head_dim=66.
    # 3D token grid 20x20x20=8000 tokens (8^3 patches over 160^3 crop), batch=2.
    # rope_dim = round(head_dim/1.5) = 44 (the eva.py formula); gives rope_channels=66=head_dim.
    _B = 2
    _NH = 12
    _N = 8000  # 20x20x20
    _HD = 66  # head_dim
    _ROPE_DIM = 44  # RotaryEmbeddingCat dim param; gives rope_channels=66
    _GRID = [20, 20, 20]

    def inputs(self, device, dtype):
        from timm.layers import RotaryEmbeddingCat

        q = torch.randn(self._B, self._NH, self._N, self._HD, device=device, dtype=dtype)
        k = torch.randn(self._B, self._NH, self._N, self._HD, device=device, dtype=dtype)
        # Build the rope embed exactly as Primus's Eva does: RotaryEmbeddingCat(...).get_embed().
        # Shape (N, rope_channels*2) = (8000, 132). Legitimately cached per fixed grid, shared
        # across all blocks, so building it here (outside the timed rotation) is faithful.
        rope = RotaryEmbeddingCat(self._ROPE_DIM, in_pixels=False, feat_shape=self._GRID, ref_feat_shape=self._GRID)
        emb = rope.get_embed().to(device=device, dtype=dtype)
        return q, k, emb

    def baseline(self, q, k, emb):
        # The op: apply_rot_embed_cat broadcasts the (N, head_dim*2) embed over the (B, NH)
        # leading dims, and here rope_channels == head_dim so every channel is rotated.
        from timm.layers.pos_embed_sincos import apply_rot_embed_cat

        return apply_rot_embed_cat(q, emb), apply_rot_embed_cat(k, emb)

    def verify(self, out, ref) -> bool:
        # The runner extracts _first() (q_out tensor) from each workload's output.
        # Compare with bf16->float32 upcast for an honest atol check.
        # bf16 fp32-accumulate vs bf16-accumulate can differ by ~1 bf16 ULP (~3e-2 at |x|~1);
        # atol=5e-2 is still honest (< 2 ULPs at bf16 scale) for this rotation-only op.
        return bool(torch.allclose(out.float(), ref.float(), atol=5e-2))
