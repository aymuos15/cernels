# config list

`reference` = the correctness/speedup reference (run as `op_eager` + `op_compile`, or as `hub` when a Hub kernel is itself the reference). `hub` / `lib` / `custom` = contenders. See [setting up baselines](guide/setting_up_baselines.md).

| config | reference | hub | lib | custom |
|---|---|---|---|---|
| [rotary](../src/configs/registry/rotary.py) | [apply_rotary_pos_emb](https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/modeling_llama.py) (transformers) | [kernels-community/rotary](https://huggingface.co/kernels-community/rotary) | — | [rope](../src/kops/registry/rope.cu) |
| [nms](../src/configs/registry/nms.py) | [torchvision.ops.nms](https://pytorch.org/vision/stable/generated/torchvision.ops.nms.html) | — | — | [nms](../src/kops/registry/nms.cu) |
| [gaussian_blur](../src/configs/registry/gaussian_blur.py) | [kornia.filters.gaussian_blur2d](https://kornia.readthedocs.io/en/latest/filters.html#kornia.filters.gaussian_blur2d) | — | — | [gaussian_blur](../src/kops/registry/gaussian_blur.cu) |
| [megablocks_moe](../src/configs/registry/megablocks_moe.py) | [kernels-community/megablocks](https://huggingface.co/kernels-community/megablocks) MoE (Hub kernel is the reference) | — | — | [moe](../src/kops/registry/moe.cu) |
| [primus_3d_rope](../src/configs/registry/primus_3d_rope.py) | [RotaryEmbeddingCat + apply_rot_embed_cat](https://github.com/MIC-DKFZ/dynamic-network-architectures/blob/main/dynamic_network_architectures/building_blocks/eva.py) (timm, as Primus uses) | — | — | [rope3d](../src/kops/registry/rope3d.cu) |
| [deformable_attention](../src/configs/registry/deformable_attention.py) | [multi_scale_deformable_attention](https://github.com/huggingface/transformers/blob/main/src/transformers/models/deformable_detr/modeling_deformable_detr.py) (transformers) | [kernels-community/deformable-detr](https://huggingface.co/kernels-community/deformable-detr) | — | [deform_attn](../src/kops/registry/deform_attn.cu) |
| [roi_align](../src/configs/registry/roi_align.py) | [torchvision.ops.roi_align](https://pytorch.org/vision/stable/generated/torchvision.ops.roi_align.html) | — | — | [roi_align](../src/kops/registry/roi_align.cu) |
| [rmsnorm](../src/configs/registry/rmsnorm.py) | [torch.nn.functional.rms_norm](https://pytorch.org/docs/stable/generated/torch.nn.functional.rms_norm.html) | — | — | [rmsnorm](../src/kops/registry/rmsnorm.cu) |
| [silu_mul](../src/configs/registry/silu_mul.py) | F.silu(gate) * up (canonical SwiGLU) | — | — | [silu_mul](../src/kops/registry/silu_mul.cu) |

## Latest results (GB10 / sm_121)

Speedups normalized to `op_eager` (the reference) = 1.00×; ✓ = verifies against the reference. `op_compile` = `torch.compile` of the reference. For `megablocks_moe` the reference is the Hub kernel, so its `hub` column is the 1.00× baseline.

| config | op_eager | op_compile | hub | lib | custom | notes |
|---|---|---|---|---|---|---|
| rotary | 1.00× | 4.90× ✓ | 2.49× ✓ | — | **5.14× ✓** | custom edges op_compile |
| nms | 1.00× | 0.98× ✓ | — | — | **1.45× ✓** | op_compile ~flat; custom beats torchvision |
| gaussian_blur | 1.00× | 0.51× ✓ | — | — | **2.58× ✓** | op_compile slower; custom ~5× vs op_compile |
| megablocks_moe | — | — | 1.00× | — | **1.31× ✓** | custom (cuBLAS Tensor-Core grouped GEMM) beats the megablocks Hub kernel |
| primus_3d_rope | 1.00× | 6.11× ✓ | — | — | **6.36× ✓** | custom edges op_compile |
| deformable_attention | 1.00× | 0.76× ✓ | 17.8× ✓ | — | **25.5× ✓** | op_compile slower; custom (1-thread/channel, float4, occupancy-tuned) beats the Hub CUDA kernel ~1.4× |
| roi_align | 1.00× | 0.90× ✓ | — | — | **1.14× ✓** | custom (bilinear sampling, fp32) beats torchvision; op_compile slower |
| rmsnorm | 1.00× | 0.92× ✓ | — | — | **1.14× ✓** | op_compile slower than eager; custom (bf16, fp32 reduce) beats compile ~1.24× |
| silu_mul | 1.00× | 1.70× ✓ | — | — | **1.74× ✓** | custom (fused silu·mul) edges op_compile |

References follow [setting up baselines](guide/setting_up_baselines.md): always a real library/Hub reference, never hand-written; the reference is only the op call, all prep in `inputs()`.
