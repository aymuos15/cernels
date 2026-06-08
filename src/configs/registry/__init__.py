"""The registry: every benchmarkable kernel config, keyed by name. Add a kernel here."""

from configs.base import Config
from configs.registry.multi_scale_deformable_attention import MultiScaleDeformableAttention
from configs.registry.gaussian_blur_2d import GaussianBlur2d
from configs.registry.gpt_oss_moe_experts import GptOssMoeExperts
from configs.registry.megablocks_moe import MegablocksMoe
from configs.registry.non_maximum_suppression import NonMaximumSuppression
from configs.registry.primus_3d_rope import Primus3dRope
from configs.registry.qwen3_next_gated_deltanet import Qwen3NextGatedDeltanet
from configs.registry.qwen3_next_gated_rmsnorm import Qwen3NextGatedRmsnorm
from configs.registry.qwen3_next_moe_experts import Qwen3NextMoeExperts
from configs.registry.rms_norm import RmsNorm
from configs.registry.roi_align import RoiAlign
from configs.registry.rotary_embedding import RotaryEmbedding
from configs.registry.sam_decomposed_rel_pos import SamDecomposedRelPos
from configs.registry.silu_and_mul import SiluAndMul

CONFIGS: dict[str, type[Config]] = {
    c.name: c
    for c in (
        RotaryEmbedding,
        NonMaximumSuppression,
        GaussianBlur2d,
        MegablocksMoe,
        GptOssMoeExperts,
        Primus3dRope,
        Qwen3NextGatedDeltanet,
        Qwen3NextGatedRmsnorm,
        Qwen3NextMoeExperts,
        MultiScaleDeformableAttention,
        RoiAlign,
        RmsNorm,
        SiluAndMul,
        SamDecomposedRelPos,
    )
}
