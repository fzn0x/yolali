"""Match one image's predictions to its ground truth at several IoU thresholds.

Follows the COCO evaluation protocol (Lin et al. 2014; pycocotools
``COCOeval.evaluateImg``), reimplemented from the protocol description:

* predictions are visited in descending score;
* each takes the not-yet-matched *regular* ground truth of its class with
  the highest IoU at or above the threshold;
* if none qualifies, it may match an *ignored* ground truth (a crowd region,
  or a box outside the current area range) and is then excluded from both
  TP and FP counts;
* crowd regions use intersection-over-prediction-area instead of IoU and may
  absorb any number of predictions.

Matching is done independently per threshold, so a prediction may be a TP
at 0.5 and an FP at 0.75.
"""

from __future__ import annotations

import torch
from torch import Tensor

from yolali.ops.boxes import box_iou

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


def _best(cand: Tensor, thr: float) -> int:
    """Index of the largest candidate at or above ``thr``, or ``-1``."""
    j = int(cand.argmax())
    return j if float(cand[j]) >= thr else -1


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

    order = torch.argsort(-pred_scores, stable=True)
    same_class = pred_classes.unsqueeze(1) == gt_classes.unsqueeze(0)  # (N, M)
    iou = box_iou(pred_boxes, gt_boxes) * same_class  # (N, M)
    if bool(crowd.any()):
        iou[:, crowd] = crowd_iou(pred_boxes, gt_boxes[crowd]) * same_class[:, crowd]

    for ti in range(t):
        thr = float(thresholds[ti])
        taken = torch.zeros(m, dtype=torch.bool)
        for k in range(n):
            i = int(order[k])
            row = iou[i]
            # Stage 1: regular ground truth that is still free.
            regular = row.masked_fill(ignored | taken, -1.0)
            j = _best(regular, thr)
            if j >= 0:
                taken[j] = True
                tp[i, ti] = True
                continue
            # Stage 2: ignored ground truth (crowd may be reused, others once).
            blocked = ~ignored | (taken & ~crowd)
            j = _best(row.masked_fill(blocked, -1.0), thr)
            if j >= 0:
                taken[j] = True
                ignore[i, ti] = True
    return tp, ignore
