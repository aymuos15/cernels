"""RoI Align (SAM3 box-prompt pooling).

Baseline: torchvision.ops.roi_align (the op IS the reference, like nms). Custom: our
fused bilinear-sampling CUDA kernel (roi_align.cu), reusing the deformable-attention
sampling machinery. No Hub lib — torchvision is the production op to beat.

Timed path: the full bilinear sampling + averaging over RoI bins.
Precomputed in inputs(): the feature map + boxes (raw tensors a caller would pass).
Semantics matched to torchvision: aligned offset 0.5, sampling grid, zero outside.
"""

import torch

from configs.base import Config
from kops.registry.roi_align import kernel as roi_align_kernel


class RoiAlign(Config):
    name = "roi_align"
    # fp32: torchvision computes the sampling-coord math in the input dtype, so fp32 makes
    # our fp32 kernel verify cleanly (fp16 coords would diverge by ~0.1 at large positions).
    dtype = torch.float32
    op = "torchvision.ops.roi_align"
    custom = staticmethod(roi_align_kernel)

    # SAM3-ish: feature map (B, C, H, W), boxes pooled to output_size x output_size.
    _B = 2
    _C = 256
    _H = 72
    _W = 72
    _num_boxes = 128
    _output_size = 7
    _spatial_scale = 1.0
    _sampling_ratio = 2
    _aligned = True

    def inputs(self, device, dtype):
        value = torch.randn(self._B, self._C, self._H, self._W, device=device, dtype=dtype)
        # boxes as (K, 5): [batch_idx, x1, y1, x2, y2], valid in-frame xyxy.
        if str(device) == "meta":
            boxes = torch.empty(self._num_boxes, 5, device=device, dtype=dtype)
            return value, boxes, self._output_size, self._spatial_scale, self._sampling_ratio, self._aligned
        torch.manual_seed(0)
        batch_idx = torch.randint(0, self._B, (self._num_boxes, 1), device=device).to(dtype)
        x1 = torch.rand(self._num_boxes, 1, device=device, dtype=dtype) * (self._W - 8)
        y1 = torch.rand(self._num_boxes, 1, device=device, dtype=dtype) * (self._H - 8)
        w = torch.rand(self._num_boxes, 1, device=device, dtype=dtype) * 8 + 1
        h = torch.rand(self._num_boxes, 1, device=device, dtype=dtype) * 8 + 1
        boxes = torch.cat([batch_idx, x1, y1, x1 + w, y1 + h], dim=1)
        return value, boxes, self._output_size, self._spatial_scale, self._sampling_ratio, self._aligned

    def baseline(self, value, boxes, output_size, spatial_scale, sampling_ratio, aligned):
        import torchvision

        return torchvision.ops.roi_align(value, boxes, output_size, spatial_scale, sampling_ratio, aligned)

    def verify(self, out, ref) -> bool:
        return bool(torch.allclose(out.float(), ref.float(), atol=1e-2, rtol=1e-2))
