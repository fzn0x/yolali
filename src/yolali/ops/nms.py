"""Class-aware non-maximum suppression for the classic one-to-many head."""

from __future__ import annotations

import torch
import torchvision
from torch import Tensor


def batched_nms(
    boxes: Tensor,
    scores: Tensor,
    classes: Tensor,
    iou_threshold: float,
    max_det: int | None = None,
) -> Tensor:
    """Suppress overlapping boxes of the same class.

    Args:
        boxes: ``(N, 4)`` xyxy pixels.
        scores: ``(N,)`` confidence.
        classes: ``(N,)`` int64 class index; boxes of different classes never
            suppress each other.
        iou_threshold: boxes with IoU above this to a kept higher-scoring box
            of the same class are removed.
        max_det: keep at most this many, highest scores first.

    Returns:
        ``(K,)`` int64 indices into the inputs, sorted by decreasing score.
    """
    if boxes.numel() == 0:
        return torch.zeros(0, dtype=torch.int64, device=boxes.device)
    keep = torchvision.ops.batched_nms(boxes.float(), scores.float(), classes, iou_threshold)
    if max_det is not None:
        keep = keep[:max_det]
    return keep
