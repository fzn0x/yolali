"""Match one image's predictions to its ground truth at several IoU thresholds.

Follows the COCO evaluation protocol (Lin et al. 2014; pycocotools
``COCOeval.evaluateImg``), reimplemented from the protocol description:

* predictions are visited in descending score, class by class;
* each takes the not-yet-matched *regular* ground truth of its class with
  the highest IoU at or above the threshold;
* if none qualifies, it may match an *ignored* ground truth (a crowd region,
  or a box outside the current area range) and is then excluded from both
  TP and FP counts;
* crowd regions use intersection-over-prediction-area instead of IoU and may
  absorb any number of predictions.

Matching is greedy over predictions but independent across thresholds, so
all thresholds are processed at once as a ``(T, M)`` "taken" table.
"""

from __future__ import annotations

import torch
from torch import Tensor

from yolali.ops.boxes import box_iou, unique_values

IOU_THRESHOLDS = torch.linspace(0.5, 0.95, 10)


def crowd_iou(pred: Tensor, crowd: Tensor) -> Tensor:
    """``(N, M)`` intersection over *prediction* area, as COCO uses for crowd regions."""
    px1, py1, px2, py2 = pred.unsqueeze(1).unbind(-1)  # (N, 1)
    cx1, cy1, cx2, cy2 = crowd.unsqueeze(0).unbind(-1)  # (1, M)
    inter = (torch.minimum(px2, cx2) - torch.maximum(px1, cx1)).clamp(min=0) * (
        torch.minimum(py2, cy2) - torch.maximum(py1, cy1)
    ).clamp(min=0)
    area = ((px2 - px1) * (py2 - py1)).clamp(min=1e-9)
    return inter / area


def _match_one_class(
    iou: Tensor, ignored: Tensor, crowd: Tensor, thresholds: Tensor
) -> tuple[Tensor, Tensor]:
    """Greedy matching for the predictions (rows, already score-sorted) of one class.

    Args:
        iou: ``(N, M)`` overlap between this class's predictions and ground truths.
        ignored: ``(M,)`` bool.
        crowd: ``(M,)`` bool (subset of ``ignored``; may be matched repeatedly).
        thresholds: ``(T,)``.

    Returns:
        ``tp (N, T)``, ``ignore (N, T)``.
    """
    n, m = iou.shape
    t = int(thresholds.shape[0])
    tp = torch.zeros(n, t, dtype=torch.bool)
    ignore = torch.zeros(n, t, dtype=torch.bool)
    taken = torch.zeros(t, m, dtype=torch.bool)
    rows = torch.arange(t)
    for k in range(n):
        row = iou[k].unsqueeze(0).expand(t, m)
        # Stage 1: free regular ground truth with the best overlap.
        regular = row.masked_fill(ignored.unsqueeze(0) | taken, -1.0)
        best_val, best_j = regular.max(1)
        hit = best_val >= thresholds
        # Stage 2 (only where stage 1 failed): ignored ground truth; crowd can be reused.
        blocked = ~ignored.unsqueeze(0) | (taken & ~crowd.unsqueeze(0))
        alt_val, alt_j = row.masked_fill(blocked, -1.0).max(1)
        alt_hit = ~hit & (alt_val >= thresholds)
        taken[rows[hit], best_j[hit]] = True
        taken[rows[alt_hit], alt_j[alt_hit]] = True
        tp[k] = hit
        ignore[k] = alt_hit
    return tp, ignore


def match_predictions(
    pred_boxes: Tensor,
    pred_scores: Tensor,
    pred_classes: Tensor,
    gt_boxes: Tensor,
    gt_classes: Tensor,
    *,
    gt_crowd: Tensor | None = None,
    gt_ignore: Tensor | None = None,
    thresholds: Tensor = IOU_THRESHOLDS,
) -> tuple[Tensor, Tensor]:
    """Greedy per-class matching.

    Args:
        pred_boxes: ``(N, 4)`` xyxy.
        pred_scores: ``(N,)``.
        pred_classes: ``(N,)`` int64.
        gt_boxes: ``(M, 4)`` xyxy.
        gt_classes: ``(M,)`` int64.
        gt_crowd: ``(M,)`` bool crowd regions (always ignored; crowd overlap formula).
        gt_ignore: ``(M,)`` bool extra ground truths to ignore (e.g. outside the
            area range); they keep the normal IoU formula and can be matched once.
        thresholds: ``(T,)`` IoU thresholds.

    Returns:
        ``tp``: ``(N, T)`` bool, true positive at each threshold.
        ``ignore``: ``(N, T)`` bool, matched an ignored ground truth (exclude
        from both TP and FP counts).
    """
    n, m, t = int(pred_boxes.shape[0]), int(gt_boxes.shape[0]), int(thresholds.shape[0])
    tp = torch.zeros(n, t, dtype=torch.bool)
    ignore = torch.zeros(n, t, dtype=torch.bool)
    if n == 0 or m == 0:
        return tp, ignore
    crowd = gt_crowd if gt_crowd is not None else torch.zeros(m, dtype=torch.bool)
    ignored = crowd | (gt_ignore if gt_ignore is not None else torch.zeros(m, dtype=torch.bool))

    for c in unique_values(gt_classes):
        p: Tensor = (pred_classes == c).nonzero(as_tuple=True)[0]
        if p.shape[0] == 0:
            continue
        p = p[torch.argsort(-pred_scores[p], stable=True)]
        g: Tensor = (gt_classes == c).nonzero(as_tuple=True)[0]
        iou = box_iou(pred_boxes[p], gt_boxes[g])  # (n_c, m_c)
        crowd_g = crowd[g]
        if bool(crowd_g.any()):
            iou[:, crowd_g] = crowd_iou(pred_boxes[p], gt_boxes[g][crowd_g])
        tp_c, ignore_c = _match_one_class(iou, ignored[g], crowd_g, thresholds)
        tp[p] = tp_c
        ignore[p] = ignore_c
    return tp, ignore
