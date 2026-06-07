# config list

| config | eager (baseline) | lib | op | custom |
|---|---|---|---|---|
| [rotary](../src/configs/registry/rotary.py) | [apply_rotary_pos_emb](https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/modeling_llama.py) (transformers) | [kernels-community/rotary](https://huggingface.co/kernels-community/rotary) | apply_rotary_transformers | [rope](../src/kops/registry/rope.cu) |
| [nms](../src/configs/registry/nms.py) | [torchvision.ops.nms](https://pytorch.org/vision/stable/generated/torchvision.ops.nms.html) | — | torchvision.ops.nms | [nms](../src/kops/registry/nms.cu) |
| [gaussian_blur](../src/configs/registry/gaussian_blur.py) | [kornia.filters.gaussian_blur2d](https://kornia.readthedocs.io/en/latest/filters.html#kornia.filters.gaussian_blur2d) | — | kornia.filters.gaussian_blur2d | [gaussian_blur](../src/kops/registry/gaussian_blur.cu) |
| [megablocks_moe](../src/configs/registry/megablocks_moe.py) | [kernels-community/megablocks](https://huggingface.co/kernels-community/megablocks) MoE (Hub kernel is the reference) | — | kernels-community/megablocks MoE | [moe](../src/kops/registry/moe.cu) |
| [primus_3d_rope](../src/configs/registry/primus_3d_rope.py) | [RotaryEmbeddingCat + apply_rot_embed_cat](https://github.com/MIC-DKFZ/dynamic-network-architectures/blob/main/dynamic_network_architectures/building_blocks/eva.py) (timm, as Primus uses) | — | timm.layers.apply_rot_embed_cat | [rope3d](../src/kops/registry/rope3d.cu) |
| [deformable_attention](../src/configs/registry/deformable_attention.py) | [multi_scale_deformable_attention](https://github.com/huggingface/transformers/blob/main/src/transformers/models/deformable_detr/modeling_deformable_detr.py) (transformers) | [kernels-community/deformable-detr](https://huggingface.co/kernels-community/deformable-detr) | multi_scale_deformable_attention | [deform_attn](../src/kops/registry/deform_attn.cu) |

## Latest results (GB10 / sm_121)

Speedups vs the eager baseline; ✓ = verifies against the baseline. `compile` = `torch.compile` of the baseline.

| config | eager | compile | lib | custom | notes |
|---|---|---|---|---|---|
| rotary | 1.00× | 4.91× ✓ | 2.46× ✓ | **5.06× ✓** | custom edges compile |
| gaussian_blur | 1.00× | 0.51× ✓ | — | **2.57× ✓** | compile slower; custom ~5× vs compile |
| megablocks_moe | 1.00× (megablocks ref) | — | — | **1.33× ✓** | custom (cuBLAS Tensor-Core grouped GEMM) beats megablocks |
| primus_3d_rope | 1.00× | 6.33× ✓ | — | **6.61× ✓** | custom edges compile |
| deformable_attention | 1.00× | 0.75× ✓ | 16.8× ✓ | **24.8× ✓** | compile slower; custom (1-thread/channel, float4, occupancy-tuned) beats upstream CUDA lib ~1.5× |

Baselines follow [setting up baselines](guide/setting_up_baselines.md): always a real library/Hub reference, never hand-written; `baseline` is only the op call, all prep in `inputs()`.
