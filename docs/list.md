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

Speedups normalized to `op_eager` (the reference) = 1.00×; ✓ = verifies against the reference. `op_compile` = `torch.compile` of the reference. For `megablocks_moe` the reference is the Hub kernel, so its `hub` column is the 1.00× baseline. Every custom kernel is a `torch.library` custom op (with `register_fake`) so it composes with `torch.compile`; the wrapper adds a few-µs eager dispatch cost — visible on the tiniest kernels (deformable, roi) — that disappears in a compiled graph.

| config | op_eager | op_compile | hub | lib | custom | notes |
|---|---|---|---|---|---|---|
| rotary | 1.00× | 4.95× ✓ | 2.49× ✓ | — | **5.07× ✓** | custom edges op_compile and the Hub kernel |
| nms | 1.00× | 0.98× ✓ | — | — | **1.41× ✓** | op_compile ~flat; custom beats torchvision |
| gaussian_blur | 1.00× | 0.50× ✓ | — | — | **2.58× ✓** | op_compile slower; custom ~5× vs op_compile |
| megablocks_moe | — | — | 1.00× | — | **1.31× ✓** | custom (cuBLAS Tensor-Core grouped GEMM) beats the megablocks Hub kernel |
| primus_3d_rope | 1.00× | 6.20× ✓ | — | — | **6.45× ✓** | custom edges op_compile |
| deformable_attention | 1.00× | 0.74× ✓ | 18.7× ✓ | — | **21.0× ✓** | op_compile slower; custom beats the Hub CUDA kernel (~1.1×); torch.library dispatch overhead on a ~30µs kernel |
| roi_align | 1.00× | 0.89× ✓ | — | — | **1.06× ✓** | custom beats torchvision; op_compile slower; dispatch overhead on a ~90µs kernel |
| rmsnorm | 1.00× | 0.94× ✓ | — | — | **1.04× ✓** | tiny op; torch.library dispatch overhead eats most of the win, but still beats op_compile ~1.1× |
| silu_mul | 1.00× | 1.69× ✓ | — | — | **1.75× ✓** | custom (fused silu·mul) edges op_compile |

References follow [setting up baselines](guide/setting_up_baselines.md): always a real library/Hub reference, never hand-written; the reference is only the op call, all prep in `inputs()`.
