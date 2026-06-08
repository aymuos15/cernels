"""non_maximum_suppression loader."""

from kops.registry._local import load


def kernel(boxes, scores, iou):
    order = scores.argsort(descending=True)
    keep = load("non_maximum_suppression").non_maximum_suppression(boxes[order].contiguous(), float(iou))
    return order[keep.to(order.device)]
