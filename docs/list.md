# config list

`reference` = the correctness/speedup reference (run as `op_eager` + `op_compile`, or as `hub` when a Hub kernel is itself the reference). `hub` / `lib` / `custom` = contenders. See [setting up baselines](guide/setting_up_baselines.md). Names follow the one-slug-per-kernel invariant in [RULES.md §1](../RULES.md).

| config | reference | hub | lib | custom |
|---|---|---|---|---|
| [rotary_embedding](../src/configs/registry/rotary_embedding.py) | [apply_rotary_pos_emb](https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/modeling_llama.py) (transformers) | [kernels-community/rotary](https://huggingface.co/kernels-community/rotary) | — | [rotary_embedding](../src/kops/rotary_embedding/) |
| [non_maximum_suppression](../src/configs/registry/non_maximum_suppression.py) | [torchvision.ops.nms](https://pytorch.org/vision/stable/generated/torchvision.ops.nms.html) | — | — | [non_maximum_suppression](../src/kops/non_maximum_suppression/) |
| [gaussian_blur_2d](../src/configs/registry/gaussian_blur_2d.py) | [kornia.filters.gaussian_blur2d](https://kornia.readthedocs.io/en/latest/filters.html#kornia.filters.gaussian_blur2d) | — | — | [gaussian_blur_2d](../src/kops/gaussian_blur_2d/) |
| [megablocks_moe](../src/configs/registry/megablocks_moe.py) | [kernels-community/megablocks](https://huggingface.co/kernels-community/megablocks) MoE (Hub kernel is the reference) | — | — | [megablocks_moe](../src/kops/megablocks_moe/) |
| [primus_3d_rope](../src/configs/registry/primus_3d_rope.py) | [RotaryEmbeddingCat + apply_rot_embed_cat](https://github.com/MIC-DKFZ/dynamic-network-architectures/blob/main/dynamic_network_architectures/building_blocks/eva.py) (timm, as Primus uses) | — | — | [primus_3d_rope](../src/kops/primus_3d_rope/) |
| [multi_scale_deformable_attention](../src/configs/registry/multi_scale_deformable_attention.py) | [multi_scale_deformable_attention](https://github.com/huggingface/transformers/blob/main/src/transformers/models/deformable_detr/modeling_deformable_detr.py) (transformers) | [kernels-community/deformable-detr](https://huggingface.co/kernels-community/deformable-detr) | — | [multi_scale_deformable_attention](../src/kops/multi_scale_deformable_attention/) |
| [roi_align](../src/configs/registry/roi_align.py) | [torchvision.ops.roi_align](https://pytorch.org/vision/stable/generated/torchvision.ops.roi_align.html) | — | — | [roi_align](../src/kops/roi_align/) |
| [rms_norm](../src/configs/registry/rms_norm.py) | [torch.nn.functional.rms_norm](https://pytorch.org/docs/stable/generated/torch.nn.functional.rms_norm.html) | — | — | [rms_norm](../src/kops/rms_norm/) |
| [silu_and_mul](../src/configs/registry/silu_and_mul.py) | F.silu(gate) * up (canonical SwiGLU) | — | — | [silu_and_mul](../src/kops/silu_and_mul/) |
| [gpt_oss_moe_experts](../src/configs/registry/gpt_oss_moe_experts.py) | [GptOssMLP](https://github.com/huggingface/transformers/blob/main/src/transformers/models/gpt_oss/modeling_gpt_oss.py) (transformers) | — | — | [gpt_oss_moe_experts](../src/kops/gpt_oss_moe_experts/) |
| [qwen3_next_moe_experts](../src/configs/registry/qwen3_next_moe_experts.py) | [Qwen3NextSparseMoeBlock](https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen3_next/modeling_qwen3_next.py) (transformers) | — | — | [qwen3_next_moe_experts](../src/kops/qwen3_next_moe_experts/) |
| [qwen3_next_gated_deltanet](../src/configs/registry/qwen3_next_gated_deltanet.py) | [torch_chunk_gated_delta_rule](https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen3_next/modeling_qwen3_next.py) (transformers) | — | — | [qwen3_next_gated_deltanet](../src/kops/qwen3_next_gated_deltanet/) |
| [qwen3_next_gated_rmsnorm](../src/configs/registry/qwen3_next_gated_rmsnorm.py) | [Qwen3NextRMSNormGated](https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen3_next/modeling_qwen3_next.py) (transformers) | — | — | [qwen3_next_gated_rmsnorm](../src/kops/qwen3_next_gated_rmsnorm/) |
| [sam_decomposed_rel_pos](../src/configs/registry/sam_decomposed_rel_pos.py) | [add_decomposed_rel_pos](https://github.com/huggingface/transformers/blob/main/src/transformers/models/sam/modeling_sam.py) (transformers) | — | — | [sam_decomposed_rel_pos](../src/kops/sam_decomposed_rel_pos/) |
| [cohere2_moe_experts](../src/configs/registry/cohere2_moe_experts.py) | [Cohere2MoeExperts](https://github.com/huggingface/transformers/blob/main/src/transformers/models/cohere2_moe/modeling_cohere2_moe.py) (transformers, North Mini Code) | — | — | [cohere2_moe_experts](../src/kops/cohere2_moe_experts/) |
| [cohere2_moe_experts_decode](../src/configs/registry/cohere2_moe_experts.py) | [Cohere2MoeExperts](https://github.com/huggingface/transformers/blob/main/src/transformers/models/cohere2_moe/modeling_cohere2_moe.py) at n_tokens=1 (decode) | — | — | [cohere2_moe_experts](../src/kops/cohere2_moe_experts/) (decode entry point) |
| [deepseek_ocr2_moe_experts](../src/configs/registry/deepseek_ocr2_moe_experts.py) | [DeepseekOcr2TextExperts](https://github.com/huggingface/transformers/blob/main/src/transformers/models/deepseek_ocr2/modeling_deepseek_ocr2.py) (transformers, DeepSeek-OCR-2) | — | — | [deepseek_ocr2_moe_experts](../src/kops/deepseek_ocr2_moe_experts/) |
| [deepseek_ocr2_moe_experts_decode](../src/configs/registry/deepseek_ocr2_moe_experts.py) | [DeepseekOcr2TextExperts](https://github.com/huggingface/transformers/blob/main/src/transformers/models/deepseek_ocr2/modeling_deepseek_ocr2.py) at n_tokens=1 (decode) | — | — | [deepseek_ocr2_moe_experts](../src/kops/deepseek_ocr2_moe_experts/) (decode entry point) |

## Latest results (GB10 / sm_121)

Speedups normalized to `op_eager` (the reference) = 1.00×; ✓ = verifies against the reference. `op_compile` = `torch.compile` of the reference. For `megablocks_moe` the reference is the Hub kernel, so its `hub` column is the 1.00× baseline. Every custom kernel is a **kernel-builder kernel** (native `TORCH_LIBRARY` op, AOT-built via nix, or kernel-builder's cmake flow on hosts without nix — see [how to add a custom kernel](guide/how_to_add_a_custom_kernel.md)). Timing: `torch.utils.benchmark` `blocked_autorange` (see [timing methodology](guide/running_benchmarks.md#timing-methodology)).

| config | op_eager | op_compile | hub | lib | custom | notes |
|---|---|---|---|---|---|---|
| rotary_embedding | 1.00× | 4.90× ✓ | 2.51× ✓ | — | **5.18× ✓** | custom edges op_compile and the Hub kernel |
| non_maximum_suppression | 1.00× | 0.98× ✓ | — | — | **2.41× ✓** | op_compile ~flat; custom beats torchvision (AOT build faster) |
| gaussian_blur_2d | 1.00× | 0.51× ✓ | — | — | **2.64× ✓** | op_compile slower; custom ~5.2× vs op_compile |
| megablocks_moe | — | — | 1.00× | — | **1.30× ✓** | custom (cuBLAS Tensor-Core grouped GEMM) beats the megablocks Hub kernel |
| primus_3d_rope | 1.00× | 6.20× ✓ | — | — | **6.60× ✓** | custom edges op_compile |
| multi_scale_deformable_attention | 1.00× | 0.71× ✓ | 19.2× ✓ | — | **27.3× ✓** | op_compile slower; custom beats the Hub CUDA kernel ~1.4× |
| roi_align | 1.00× | 0.81× ✓ | — | — | **1.12× ✓** | custom beats torchvision; op_compile slower |
| rms_norm | 1.00× | 0.91× ✓ | — | — | **1.12× ✓** | tiny op; custom beats op_compile ~1.2× |
| silu_and_mul | 1.00× | 1.68× ✓ | — | — | **1.75× ✓** | custom ≈ op_compile (ties; compile fuses this cheap elementwise as well) |
| gpt_oss_moe_experts | 1.00× | 1.10× ✓ | — | — | **1.40× ✓** | grouped GEMM + clamped-limited SwiGLU; 1.27× vs op_compile |
| qwen3_next_moe_experts | 1.00× | 0.98× ✓ | — | — | **2.69× ✓** | 512 experts top-10 + shared expert; 2.75× vs op_compile |
| qwen3_next_gated_deltanet | 1.00× | 1.40× ✓ | — | — | **1.93× ✓** | chunked delta-rule linear attention; 1.38× vs op_compile |
| qwen3_next_gated_rmsnorm | 1.00× | 10.5× ✓ | — | — | **11.1× ✓** | fused gated RMSNorm; ties op_compile (bandwidth-bound single pass) |
| sam_decomposed_rel_pos | 1.00× | 0.99× ✓ | — | — | **23.4× ✓** | decomposed rel-pos attention bias; 23.6× vs op_compile |
| cohere2_moe_experts | 1.00× | 1.01× ✓ | — | — | **2.13× ✓** | North Mini Code 128-expert top-8 grouped GEMM; 2.11× vs op_compile |
| cohere2_moe_experts_decode | 1.00× | 0.98× ✓ | — | — | **3.50× ✓** | fused top-8 gather-GEMV at n_tokens=1; 3.58× vs op_compile, near the weight-traffic floor |
| deepseek_ocr2_moe_experts | 1.00× | 0.96× ✓ | — | — | **2.17× ✓** | DeepSeek-OCR-2 64-expert top-6 grouped GEMM; 2.26× vs op_compile |
| deepseek_ocr2_moe_experts_decode | 1.00× | 0.94× ✓ | — | — | **4.10× ✓** | fused top-6 gather-GEMV at n_tokens=1; 4.35× vs op_compile |

References follow [setting up baselines](guide/setting_up_baselines.md): always a real library/Hub reference, never hand-written; the reference is only the op call, all prep in `inputs()`.
