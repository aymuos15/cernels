"""Multi-scale deformable attention (Deformable DETR family).

Reference: transformers' MultiScaleDeformableAttention.forward (pure-torch grid_sample, op_eager).
Hub:       Hub kernel kernels-community/deformable-detr (ms_deform_attn_forward, fused CUDA).
Custom:    our own fused CUDA kernel (deform_attn.cu) — bilinear sample + weight + fp32 accumulate
           in a single pass, no per-level intermediate tensors.

Timed path for all workloads: the full bilinear-sampling + weighted reduction.
Precomputed in inputs(): value, spatial_shapes, level_start_index, sampling_locations,
                         attention_weights — all raw tensors the real caller would pass.
grid_sample semantics: align_corners=False, padding_mode=zeros; px = loc_x*W - 0.5 (custom kernel
matches torch's convention exactly via this formula, tested to atol~1e-5 vs baseline).
"""

from typing import Any

import torch

from configs.base import Config
from kops.registry.deform_attn import kernel as deform_attn_kernel


class DeformableAttention(Config):
    name = "deformable_attention"
    dtype = torch.float32
    op = "MultiScaleDeformableAttention.forward"
    use_compile = True

    # Hub contender: ms_deform_attn_forward from kernels-community/deformable-detr
    _hub_kernel: Any = None

    custom = staticmethod(deform_attn_kernel)

    # Deformable-DETR defaults
    _B = 2
    _n_heads = 8
    _hidden = 32  # hidden per head; model dim = 256
    _n_levels = 4
    _n_points = 4
    _num_queries = 300
    _spatial_shapes_list = [(92, 92), (46, 46), (23, 23), (12, 12)]
    _im2col_step = 64

    def inputs(self, device, dtype):
        spatial_shapes_list = self._spatial_shapes_list
        sp = torch.tensor(spatial_shapes_list, dtype=torch.long, device=device)
        L_total = sum(h * w for h, w in spatial_shapes_list)
        lsi = torch.zeros(self._n_levels, dtype=torch.long, device=device)
        if device != torch.device("meta"):
            lsi[1:] = (sp[:, 0] * sp[:, 1]).cumsum(0)[:-1]

        value = torch.randn(self._B, L_total, self._n_heads, self._hidden, device=device, dtype=dtype)
        sampling_locations = torch.rand(
            self._B,
            self._num_queries,
            self._n_heads,
            self._n_levels,
            self._n_points,
            2,
            device=device,
            dtype=dtype,
        )
        attention_weights = torch.rand(
            self._B,
            self._num_queries,
            self._n_heads,
            self._n_levels,
            self._n_points,
            device=device,
            dtype=dtype,
        )
        if device != torch.device("meta"):
            # softmax-normalize over levels*points
            attention_weights = attention_weights / attention_weights.sum(dim=[-1, -2], keepdim=True)

        return value, sp, spatial_shapes_list, lsi, sampling_locations, attention_weights, self._im2col_step

    def baseline(self, value, sp, spatial_shapes_list, lsi, sampling_locations, attention_weights, im2col_step):
        # transformers' pure-torch grid_sample reference — the op_eager workload and correctness reference
        from transformers.models.deformable_detr.modeling_deformable_detr import MultiScaleDeformableAttention

        m = MultiScaleDeformableAttention.__new__(MultiScaleDeformableAttention)
        torch.nn.Module.__init__(m)
        return m.forward(value, sp, spatial_shapes_list, lsi, sampling_locations, attention_weights, im2col_step)

    def hub(self, value, sp, spatial_shapes_list, lsi, sampling_locations, attention_weights, im2col_step):
        # Hub kernel: fused CUDA ms_deform_attn_forward
        kernel = self._hub_kernel
        if kernel is None:
            from kernels import get_kernel

            kernel = get_kernel("kernels-community/deformable-detr", version=1)
            type(self)._hub_kernel = kernel
        return kernel.ms_deform_attn_forward(value, sp, lsi, sampling_locations, attention_weights, im2col_step)

    def verify(self, out, ref) -> bool:
        return bool(torch.allclose(out, ref, atol=1e-3))
