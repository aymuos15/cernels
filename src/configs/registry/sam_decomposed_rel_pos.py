"""SAM decomposed relative-position attention bias.

Reference: transformers' `SamVisionAttention.get_decomposed_rel_pos` + the broadcast add into
the pre-softmax logits (the `add_decomposed_rel_pos` op, as the model uses it inside every SAM
ViT attention block). Called directly off a real `SamVisionAttention` instance — never
hand-written (see docs/guide/setting_up_baselines.md).

The op: from the (B, q_h*q_w, c) query and the learned `rel_pos_h`/`rel_pos_w` tables it
  - get_rel_pos: interpolates each table to 2*max(q,k)-1 and gathers by relative coordinate
    -> Rh (q_h, k_h, c), Rw (q_w, k_w, c),
  - reshapes query to (B, q_h, q_w, c),
  - rel_h = einsum("bhwc,hkc->bhwk"), rel_w = einsum("bhwc,wkc->bhwk"),
  - adds rel_h[...,:,None] + rel_w[...,None,:] into attn.view(B, q_h, q_w, k_h, k_w).

Timed path (inside baseline/custom): the get_rel_pos interpolation+gather, both einsums, and
the broadcast bias add — the whole fusable core, exactly as the model calls it per forward.
Precomputed in inputs(): only the raw tensors (attn logits, query, the two learned tables) and
the q_size/k_size ints. The gathered Rh/Rw are deliberately NOT precomputed — doing so would
trivialize the op (see docs/guide/correctness.md).

Realistic SAM ViT-B windowed shape: 12 heads, head_dim 64, window 14 -> q_h=q_w=k_h=k_w=14,
so attn is (12, 196, 196), query is (12, 196, 64), tables are (27, 64).
"""

import torch

from configs.base import Config
from kops.registry.sam_decomposed_rel_pos import kernel as sam_kernel


def _attn_module(device):
    from transformers.models.sam.configuration_sam import SamVisionConfig
    from transformers.models.sam.modeling_sam import SamVisionAttention

    config = SamVisionConfig()  # ViT-B: hidden_size 768, 12 heads -> head_dim 64, window 14
    mod = SamVisionAttention(config, window_size=config.window_size).to(device)
    return mod


class SamDecomposedRelPos(Config):
    name = "sam_decomposed_rel_pos"
    dtype = torch.float16
    op = "transformers SamVisionAttention.get_decomposed_rel_pos"
    custom = staticmethod(sam_kernel)

    _NH = 12  # batch_size * num_attention_heads (batch_size = 1)
    _HD = 64  # head_dim
    _W = 14  # window size -> q_h = q_w = k_h = k_w
    _L = 27  # 2 * window - 1

    def inputs(self, device, dtype):
        nh, hd, w = self._NH, self._HD, self._W
        attn = torch.randn(nh, w * w, w * w, device=device, dtype=dtype)
        query = torch.randn(nh, w * w, hd, device=device, dtype=dtype)
        rel_pos_h = torch.randn(self._L, hd, device=device, dtype=dtype)
        rel_pos_w = torch.randn(self._L, hd, device=device, dtype=dtype)
        return attn, query, rel_pos_h, rel_pos_w, w, w

    def baseline(self, attn, query, rel_pos_h, rel_pos_w, q_w, k_w):
        mod = _attn_module(attn.device)
        bias = mod.get_decomposed_rel_pos(query, rel_pos_h, rel_pos_w, (q_w, q_w), (k_w, k_w))
        return attn + bias.reshape_as(attn)

    def verify(self, out, ref) -> bool:
        # fp16 inputs; the bias dominates and logits reach magnitude ~57, where one fp16 ULP is
        # ~0.06. The fp16 eager einsum, the compiled einsum, and our fp32-accumulating kernel all
        # disagree by ~1 ULP — unavoidable fp16 rounding (op_compile itself differs from op_eager
        # by the same 0.031). A relative bar (rtol 2e-3 -> ~0.11 at 57) honestly absorbs that
        # rounding while staying tight; the small atol covers values near zero.
        return bool(torch.allclose(out.float(), ref.float(), atol=2e-2, rtol=2e-3))
