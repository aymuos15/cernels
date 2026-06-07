"""SAM decomposed-rel-pos loader — kernel-builder kernel under src/kops/sam_decomposed_rel_pos/.

The fusable core (the two einsums + broadcast bias add) runs in the CUDA op. The get_rel_pos
interpolation+gather that produces Rh/Rw stays in Python here — and since this loader IS the
timed `custom` callable, that work is still inside the timed path (it is not precomputed in
inputs()). We reuse transformers' own `SamVisionAttention.get_rel_pos` so the interpolation is
the real reference math, never hand-written.
"""

from pathlib import Path
from typing import Any

_mod: Any = None
_attn: Any = None
_REPO = Path(__file__).resolve().parents[1] / "sam_decomposed_rel_pos"


def _module():
    global _mod
    if _mod is None:
        from kernels import get_local_kernel

        _mod = get_local_kernel(_REPO)
    return _mod


def _attn_mod(device):
    global _attn
    if _attn is None:
        from transformers.models.sam.configuration_sam import SamVisionConfig
        from transformers.models.sam.modeling_sam import SamVisionAttention

        config = SamVisionConfig()
        _attn = SamVisionAttention(config, window_size=config.window_size).to(device)
    return _attn


def kernel(attn, query, rel_pos_h, rel_pos_w, q_w, k_w):
    # get_rel_pos: real transformers interpolation + relative-coordinate gather (timed path).
    m = _attn_mod(attn.device)
    Rh = m.get_rel_pos(q_w, k_w, rel_pos_h)  # (q_h, k_h, C)
    Rw = m.get_rel_pos(q_w, k_w, rel_pos_w)  # (q_w, k_w, C)
    # query stays (B, q_h*q_w, C); the kernel derives q_h/q_w/k_h/k_w from Rh/Rw shapes.
    return _module().sam_decomposed_rel_pos(query, Rh.contiguous(), Rw.contiguous(), attn)
