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

## Latest results (GB10 / sm_121)

Speedups normalized to `op_eager` (the reference) = 1.00×; ✓ = verifies against the reference. `op_compile` = `torch.compile` of the reference. For `megablocks_moe` the reference is the Hub kernel, so its `hub` column is the 1.00× baseline. Every custom kernel is a **kernel-builder kernel** (native `TORCH_LIBRARY` op, AOT-built via nix — see [how to add a custom kernel](guide/how_to_add_a_custom_kernel.md)).

| config | op_eager | op_compile | hub | lib | custom | notes |
|---|---|---|---|---|---|---|
| rotary_embedding | 1.00× | 4.85× ✓ | 2.54× ✓ | — | **5.17× ✓** | custom edges op_compile and the Hub kernel |
| non_maximum_suppression | 1.00× | 0.98× ✓ | — | — | **2.47× ✓** | op_compile ~flat; custom beats torchvision (AOT build faster) |
| gaussian_blur_2d | 1.00× | 0.55× ✓ | — | — | **2.65× ✓** | op_compile slower; custom ~5× vs op_compile |
| megablocks_moe | — | — | 1.00× | — | **1.26× ✓** | custom (cuBLAS Tensor-Core grouped GEMM) beats the megablocks Hub kernel |
| primus_3d_rope | 1.00× | 5.99× ✓ | — | — | **6.33× ✓** | custom edges op_compile |
| multi_scale_deformable_attention | 1.00× | 0.70× ✓ | 17.7× ✓ | — | **24.5× ✓** | op_compile slower; custom beats the Hub CUDA kernel ~1.4× |
| roi_align | 1.00× | 0.87× ✓ | — | — | **1.15× ✓** | custom beats torchvision; op_compile slower |
| rms_norm | 1.00× | 0.94× ✓ | — | — | **1.15× ✓** | tiny op; custom beats op_compile ~1.2× |
| silu_and_mul | 1.00× | 1.66× ✓ | — | — | **1.64× ✓** | custom ≈ op_compile (ties; compile fuses this cheap elementwise as well) |
| gpt_oss_moe_experts | 1.00× | 1.10× ✓ | — | — | **1.41× ✓** | grouped GEMM + clamped-limited SwiGLU; 1.28× vs op_compile |
| qwen3_next_moe_experts | 1.00× | 0.99× ✓ | — | — | **2.67× ✓** | 512 experts top-10 + shared expert; 2.70× vs op_compile |
| qwen3_next_gated_deltanet | 1.00× | 1.44× ✓ | — | — | **1.94× ✓** | chunked delta-rule linear attention; 1.35× vs op_compile |
| qwen3_next_gated_rmsnorm | 1.00× | 10.9× ✓ | — | — | **11.1× ✓** | fused gated RMSNorm; ties op_compile (bandwidth-bound single pass) |

References follow [setting up baselines](guide/setting_up_baselines.md): always a real library/Hub reference, never hand-written; the reference is only the op call, all prep in `inputs()`.
