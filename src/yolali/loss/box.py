"""Box regression losses for the anchors an assigner marked positive."""

from __future__ import annotations

import torch.nn.functional as F  # noqa: N812 - torch convention
from torch import Tensor

from yolali.ops.boxes import IoUKind, elementwise_iou


def iou_loss(pred: Tensor, target: Tensor, weight: Tensor, kind: IoUKind = "ciou") -> Tensor:
    """``sum_i weight_i * (1 - IoU_kind(pred_i, target_i))``.

    Args:
        pred: ``(N, 4)`` xyxy pixels.
        target: ``(N, 4)`` xyxy pixels.
        weight: ``(N,)`` per-box weight (the anchor's total soft target).
        kind: IoU variant.
    """
    return ((1.0 - elementwise_iou(pred, target, kind)) * weight).sum()


def dfl_loss(pred_logits: Tensor, target: Tensor, weight: Tensor, reg_max: int) -> Tensor:
    """Distribution Focal Loss (Li et al., NeurIPS 2020, arXiv:2006.04388).

    Each continuous side distance ``t`` (in stride units) is represented by
    the two integer bins that bracket it. The loss is cross-entropy against
    the left bin weighted by ``ceil(t) - t`` plus cross-entropy against the
    right bin weighted by ``t - floor(t)``, averaged over the four sides.

    Args:
        pred_logits: ``(N, 4 * reg_max)`` side-major bin logits.
        target: ``(N, 4)`` distances in stride units, clamped internally to
            ``[0, reg_max - 1 - 0.01]`` so the right bin always exists.
        weight: ``(N,)`` per-anchor weight.
        reg_max: bins per side.

    Returns:
        ``sum_i weight_i * mean_over_sides(ce)``.
    """
    n = target.shape[0]
    if n == 0:
        return pred_logits.sum() * 0.0
    target = target.clamp(0, reg_max - 1 - 0.01)
    left = target.floor().long()  # (N, 4)
    right = left + 1
    w_left = (right.to(target.dtype) - target).reshape(-1)  # (N*4,)
    w_right = 1.0 - w_left
    logits = pred_logits.reshape(-1, reg_max)  # (N*4, reg_max)
    ce_left = F.cross_entropy(logits, left.reshape(-1), reduction="none")
    ce_right = F.cross_entropy(logits, right.reshape(-1), reduction="none")
    per_side = ce_left * w_left + ce_right * w_right  # (N*4,)
    per_anchor = per_side.reshape(n, 4).mean(-1)  # (N,)
    return (per_anchor * weight).sum()
