"""Average precision the COCO way: 101-point interpolation, per class, per IoU.

Area ranges and ``maxDets`` follow pycocotools defaults so numbers are
comparable with published results: ``small < 32²``, ``32² ≤ medium ≤ 96²``,
``large > 96²`` by annotation area (segmentation area for COCO, box area
otherwise); at most 100 detections per image *and class* are scored.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt
import torch
from torch import Tensor

from yolali.eval.matching import IOU_THRESHOLDS, match_predictions

RECALL_POINTS = np.linspace(0.0, 1.0, 101)
AREA_RANGES: dict[str, tuple[float, float]] = {
    "all": (0.0, 1e5**2),
    "small": (0.0, 32.0**2),
    "medium": (32.0**2, 96.0**2),
    "large": (96.0**2, 1e5**2),
}

BoolArray = npt.NDArray[np.bool_]
FloatArray = npt.NDArray[np.float64]


def average_precision(tp: BoolArray, scores: FloatArray, n_gt: int) -> float:
    """COCO AP for one class and one IoU threshold.

    Args:
        tp: ``(N,)`` true-positive flags of every scored prediction.
        scores: ``(N,)`` prediction confidences.
        n_gt: number of regular (non-ignored) ground-truth boxes of this class.

    Returns:
        Mean of precision sampled at 101 recall points ``0, 0.01, ..., 1``,
        where precision is first made monotonically non-increasing from the
        right (the ``pycocotools`` envelope). ``-1`` when ``n_gt == 0``.
    """
    if n_gt == 0:
        return -1.0
    if tp.shape[0] == 0:
        return 0.0
    order = np.argsort(-scores, kind="stable")
    tp_sorted = tp[order].astype(np.float64)
    cum_tp = np.cumsum(tp_sorted)
    cum_fp = np.cumsum(1.0 - tp_sorted)
    recall = cum_tp / n_gt
    precision = cum_tp / np.maximum(cum_tp + cum_fp, np.finfo(np.float64).eps)
    for i in range(len(precision) - 2, -1, -1):
        precision[i] = max(precision[i], precision[i + 1])
    idx = np.searchsorted(recall, RECALL_POINTS, side="left")
    sampled = np.where(idx < len(precision), precision[np.minimum(idx, len(precision) - 1)], 0.0)
    return float(sampled.mean())


@dataclass
class DetectionMetrics:
    """Summary numbers; ``-1`` means undefined (no ground truth of that kind)."""

    map50_95: float
    map50: float
    map75: float
    map_small: float
    map_medium: float
    map_large: float
    per_class_ap50_95: dict[int, float] = field(default_factory=dict[int, float])

    def as_dict(self) -> dict[str, float]:
        """Flat mapping for loggers."""
        return {
            "mAP50-95": self.map50_95,
            "mAP50": self.map50,
            "mAP75": self.map75,
            "mAP_small": self.map_small,
            "mAP_medium": self.map_medium,
            "mAP_large": self.map_large,
        }


@dataclass
class _ClassStore:
    scores: list[float] = field(default_factory=list[float])
    tp: list[BoolArray] = field(default_factory=list[BoolArray])  # each (T,)
    ignore: list[BoolArray] = field(default_factory=list[BoolArray])  # each (T,)
    n_gt: int = 0


def _top_per_class(scores: Tensor, classes: Tensor, max_det: int) -> Tensor:
    """Bool mask keeping the ``max_det`` highest-scoring predictions of each class."""
    n = int(scores.shape[0])
    keep = torch.ones(n, dtype=torch.bool)
    if n == 0:
        return keep
    # Sort by (class, -score); the rank of each prediction inside its class is
    # its position minus the position where that class starts.
    order = torch.argsort(classes.double() * 2.0 - scores.double().clamp(0, 1), stable=True)
    sorted_classes = classes[order]
    starts = torch.zeros(n, dtype=torch.int64)
    starts[1:] = (sorted_classes[1:] != sorted_classes[:-1]).cumsum(0)
    first_index = torch.zeros(int(starts[-1]) + 1, dtype=torch.int64)
    change = torch.ones(n, dtype=torch.bool)
    change[1:] = sorted_classes[1:] != sorted_classes[:-1]
    first_index[starts[change]] = change.nonzero(as_tuple=True)[0]
    rank = torch.arange(n) - first_index[starts]
    keep[order[rank >= max_det]] = False
    return keep


class MetricAccumulator:
    """Collect per-image matches, then compute :class:`DetectionMetrics`.

    Args:
        num_classes: ``nc``.
        max_det: detections kept per image and class (highest scores first).
        thresholds: IoU thresholds; default is the COCO ``0.5:0.05:0.95``.
    """

    def __init__(
        self, num_classes: int, max_det: int = 100, thresholds: Tensor = IOU_THRESHOLDS
    ) -> None:
        self.num_classes = num_classes
        self.max_det = max_det
        self.thresholds = thresholds
        self.stores: dict[str, dict[int, _ClassStore]] = {
            area: {c: _ClassStore() for c in range(num_classes)} for area in AREA_RANGES
        }

    def add_image(
        self,
        pred_boxes: Tensor,
        pred_scores: Tensor,
        pred_classes: Tensor,
        gt_boxes: Tensor,
        gt_classes: Tensor,
        *,
        gt_crowd: Tensor | None = None,
        gt_areas: Tensor | None = None,
    ) -> None:
        """Score one image. All boxes in original-image pixels.

        Args:
            pred_boxes: ``(N, 4)``.
            pred_scores: ``(N,)``.
            pred_classes: ``(N,)``.
            gt_boxes: ``(M, 4)``.
            gt_classes: ``(M,)``.
            gt_crowd: ``(M,)`` bool.
            gt_areas: ``(M,)`` annotation areas for the size buckets; defaults
                to box areas.
        """
        keep = _top_per_class(pred_scores, pred_classes, self.max_det)
        pred_boxes, pred_scores, pred_classes = (
            pred_boxes[keep],
            pred_scores[keep],
            pred_classes[keep],
        )
        m = int(gt_boxes.shape[0])
        crowd = gt_crowd if gt_crowd is not None else torch.zeros(m, dtype=torch.bool)
        pred_area = (pred_boxes[:, 2] - pred_boxes[:, 0]) * (pred_boxes[:, 3] - pred_boxes[:, 1])
        if gt_areas is None:
            gt_areas = (gt_boxes[:, 2] - gt_boxes[:, 0]) * (gt_boxes[:, 3] - gt_boxes[:, 1])

        for area, (lo, hi) in AREA_RANGES.items():
            gt_in = (gt_areas >= lo) & (gt_areas <= hi)
            tp, ignore = match_predictions(
                pred_boxes,
                pred_scores,
                pred_classes,
                gt_boxes,
                gt_classes,
                gt_crowd=crowd,
                gt_ignore=~gt_in,
                thresholds=self.thresholds,
            )
            pred_in = (pred_area >= lo) & (pred_area <= hi)
            # An unmatched prediction outside the range is ignored, not a false positive.
            ignore = ignore | (~pred_in.unsqueeze(1) & ~tp)
            for c in range(self.num_classes):
                store = self.stores[area][c]
                store.n_gt += int(((gt_classes == c) & gt_in & ~crowd).sum())
                sel = pred_classes == c
                if bool(sel.any()):
                    store.scores.extend(pred_scores[sel].tolist())
                    store.tp.extend(tp[sel].numpy())
                    store.ignore.extend(ignore[sel].numpy())

    def _ap_table(self, area: str) -> FloatArray:
        """``(T, nc)`` AP per threshold and class; ``-1`` where undefined."""
        t = int(self.thresholds.shape[0])
        table: FloatArray = np.full((t, self.num_classes), -1.0)
        for c, store in self.stores[area].items():
            if store.n_gt == 0:
                continue
            scores: FloatArray = np.asarray(store.scores, dtype=np.float64)
            tp: BoolArray = np.asarray(store.tp, dtype=np.bool_).reshape(-1, t)
            ignore: BoolArray = np.asarray(store.ignore, dtype=np.bool_).reshape(-1, t)
            for ti in range(t):
                keep = ~ignore[:, ti]
                table[ti, c] = average_precision(tp[keep, ti], scores[keep], store.n_gt)
        return table

    @staticmethod
    def _mean(values: FloatArray) -> float:
        valid = values[values > -1]
        return float(valid.mean()) if valid.size else -1.0

    def compute(self) -> DetectionMetrics:
        """Reduce everything collected so far."""
        table = self._ap_table("all")
        per_class: dict[int, float] = {
            c: self._mean(table[:, c])
            for c in range(self.num_classes)
            if bool((table[:, c] > -1).any())
        }
        return DetectionMetrics(
            map50_95=self._mean(table),
            map50=self._mean(table[0]),
            map75=self._mean(table[5]) if table.shape[0] > 5 else -1.0,
            map_small=self._mean(self._ap_table("small")),
            map_medium=self._mean(self._ap_table("medium")),
            map_large=self._mean(self._ap_table("large")),
            per_class_ap50_95=per_class,
        )
