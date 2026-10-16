"""Task-aligned one-to-many label assignment.

Follows the alignment metric of TOOD (Feng et al., ICCV 2021,
arXiv:2108.07755) as used by the YOLOv8/YOLOv9 lineage: an anchor is a good
candidate for a ground-truth box when it is *inside* the box, its predicted
box overlaps the ground truth well, and it already scores the right class.

Written to be read top to bottom; each step of :meth:`TaskAlignedAssigner.forward`
produces one named tensor whose shape is in the comment beside it.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F  # noqa: N812 - torch convention
from torch import Tensor, nn

from yolali.ops.boxes import broadcast_iou
from yolali.types import Targets


@dataclass(frozen=True)
class Assignment:
    """Per-anchor training targets produced by an assigner.

    Attributes:
        boxes: ``(B, A, 4)`` xyxy pixel box each anchor must regress to.
            Zeros where ``fg_mask`` is ``False``.
        scores: ``(B, A, nc)`` soft classification targets in ``[0, 1]``.
            Zero rows where ``fg_mask`` is ``False``.
        fg_mask: ``(B, A)`` bool, ``True`` for anchors assigned to a box.
        gt_idx: ``(B, A)`` int64 index into the ``M`` axis of the targets.
            Zero where ``fg_mask`` is ``False``; always check the mask first.
    """

    boxes: Tensor
    scores: Tensor
    fg_mask: Tensor
    gt_idx: Tensor


def anchors_inside_boxes(anchors: Tensor, boxes: Tensor, eps: float = 1e-9) -> Tensor:
    """``(A, 2)`` anchors, ``(B, M, 4)`` boxes -> ``(B, M, A)`` bool "anchor is inside box"."""
    ax, ay = anchors[:, 0], anchors[:, 1]  # (A,)
    x1, y1, x2, y2 = boxes.unbind(-1)  # each (B, M)
    # Broadcast (B, M, 1) against (A,) -> (B, M, A).
    left = ax - x1.unsqueeze(-1)
    top = ay - y1.unsqueeze(-1)
    right = x2.unsqueeze(-1) - ax
    bottom = y2.unsqueeze(-1) - ay
    return torch.stack((left, top, right, bottom), dim=-1).amin(-1) > eps


class TaskAlignedAssigner(nn.Module):
    """Pick the ``topk`` best-aligned anchors inside each ground-truth box.

    Args:
        topk: candidates per ground truth before conflict resolution.
        alpha: exponent on the predicted class score in the alignment metric.
        beta: exponent on the IoU in the alignment metric.
        eps: numerical floor.

    The metric is ``score[gt_class] ** alpha * iou ** beta``. When an anchor is
    a candidate for several boxes, the box with the highest IoU wins
    (``docs/design.md`` D1). The soft classification target for a positive
    anchor is its metric rescaled so the best anchor of each box gets that
    box's best IoU.
    """

    def __init__(
        self, topk: int = 10, alpha: float = 0.5, beta: float = 6.0, eps: float = 1e-9
    ) -> None:
        super().__init__()
        if topk < 1:
            raise ValueError("topk must be >= 1")
        self.topk = topk
        self.alpha = alpha
        self.beta = beta
        self.eps = eps

    @torch.no_grad()
    def forward(
        self, pred_scores: Tensor, pred_boxes: Tensor, anchors: Tensor, targets: Targets
    ) -> Assignment:
        """Assign anchors to ground truth.

        Args:
            pred_scores: ``(B, A, nc)`` class probabilities after sigmoid.
            pred_boxes: ``(B, A, 4)`` decoded xyxy pixel boxes.
            anchors: ``(A, 2)`` anchor centres in pixels.
            targets: padded ground truth.
        """
        batch, num_anchors, num_classes = pred_scores.shape
        num_gt = targets.max_boxes
        gt_mask = targets.mask.unsqueeze(-1)  # (B, M, 1)

        # 1. Which anchors sit inside which ground-truth box.
        in_gt = anchors_inside_boxes(anchors, targets.boxes, self.eps) & gt_mask  # (B, M, A)

        # 2. IoU between every ground truth and every predicted box.
        iou = broadcast_iou(targets.boxes.unsqueeze(2), pred_boxes.unsqueeze(1), "iou")  # (B, M, A)
        iou = iou.clamp_(min=0) * in_gt

        # 3. Predicted probability of each ground truth's class at every anchor.
        cls_index = targets.classes.unsqueeze(-1).expand(-1, -1, num_anchors)  # (B, M, A)
        gt_scores = pred_scores.permute(0, 2, 1).gather(1, cls_index)  # (B, M, A)

        # 4. Alignment metric, zero outside the box.
        metric = gt_scores.pow(self.alpha) * iou.pow(self.beta) * in_gt  # (B, M, A)

        # 5. Top-k anchors per ground truth become candidates.
        k = min(self.topk, num_anchors)
        top_vals, top_idx = metric.topk(k, dim=-1)  # (B, M, k)
        candidate = torch.zeros_like(metric, dtype=torch.bool)
        candidate.scatter_(-1, top_idx, top_vals > self.eps)
        positive = candidate & in_gt  # (B, M, A)

        # 6. An anchor claimed by several boxes goes to the one with highest IoU.
        claims = positive.sum(1)  # (B, A)
        if bool((claims > 1).any()):
            best_gt = (iou * positive).argmax(1)  # (B, A)
            exclusive = F.one_hot(best_gt, num_gt).permute(0, 2, 1).bool()  # (B, M, A)
            positive = torch.where((claims > 1).unsqueeze(1), exclusive & positive, positive)

        fg_mask = positive.any(1)  # (B, A)
        gt_idx = positive.to(torch.int64).argmax(1)  # (B, A), 0 where background

        # 7. Gather the box and class each positive anchor is responsible for.
        flat_idx = gt_idx + torch.arange(batch, device=gt_idx.device).unsqueeze(1) * num_gt
        target_boxes = targets.boxes.reshape(-1, 4)[flat_idx] * fg_mask.unsqueeze(-1)  # (B, A, 4)
        target_cls = targets.classes.reshape(-1)[flat_idx]  # (B, A)
        one_hot = F.one_hot(target_cls, num_classes).to(pred_scores.dtype)  # (B, A, nc)
        one_hot = one_hot * fg_mask.unsqueeze(-1)

        # 8. Soft targets: metric normalised per box so its best anchor gets its best IoU.
        pos_metric = metric * positive  # (B, M, A)
        max_metric = pos_metric.amax(-1, keepdim=True)  # (B, M, 1)
        max_iou = (iou * positive).amax(-1, keepdim=True)  # (B, M, 1)
        norm = (pos_metric * max_iou / max_metric.clamp(min=self.eps)).amax(1)  # (B, A)
        scores = one_hot * norm.unsqueeze(-1)

        return Assignment(target_boxes, scores, fg_mask, gt_idx)
