import torch

from configs.base import Config
from kops.registry.non_maximum_suppression import kernel as nms_kernel


class NonMaximumSuppression(Config):
    name = "non_maximum_suppression"
    dtype = torch.float32
    op = "torchvision.ops.nms"
    iou = 0.5
    custom = staticmethod(nms_kernel)

    def inputs(self, device, dtype):
        n = 4096
        xy = torch.rand(n, 2, device=device, dtype=dtype) * 1000
        wh = torch.rand(n, 2, device=device, dtype=dtype) * 200 + 10
        boxes = torch.cat([xy, xy + wh], dim=1)
        scores = torch.rand(n, device=device, dtype=dtype)
        return boxes, scores, self.iou

    def baseline(self, boxes, scores, iou):
        import torchvision

        return torchvision.ops.nms(boxes, scores, iou)

    def verify(self, out, ref):  # index sets, not allclose
        return set(out.tolist()) == set(ref.tolist())
