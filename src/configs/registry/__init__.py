"""The registry: every benchmarkable kernel config, keyed by name. Add a kernel here."""

from configs.base import Config
from configs.registry.deformable_attention import DeformableAttention
from configs.registry.gaussian_blur import GaussianBlur
from configs.registry.gpt_oss_moe import GptOssMoE
from configs.registry.megablocks_moe import MegablocksMoE
from configs.registry.nms import NMS
from configs.registry.primus_3d_rope import Primus3DRope
from configs.registry.qwen3_next_gated_deltanet import Qwen3NextGatedDeltaNet
from configs.registry.qwen3_next_gated_rmsnorm import Qwen3NextGatedRMSNorm
from configs.registry.qwen3_next_moe import Qwen3NextMoE
from configs.registry.rmsnorm import RMSNorm
from configs.registry.roi_align import RoIAlign
from configs.registry.rotary import Rotary
from configs.registry.sam_decomposed_rel_pos import SamDecomposedRelPos
from configs.registry.silu_mul import SiluMul

CONFIGS: dict[str, type[Config]] = {
    c.name: c
    for c in (
        Rotary,
        NMS,
        GaussianBlur,
        MegablocksMoE,
        GptOssMoE,
        Primus3DRope,
        Qwen3NextGatedDeltaNet,
        Qwen3NextGatedRMSNorm,
        Qwen3NextMoE,
        DeformableAttention,
        RoIAlign,
        RMSNorm,
        SiluMul,
        SamDecomposedRelPos,
    )
}
