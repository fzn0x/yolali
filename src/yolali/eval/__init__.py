"""COCO-protocol detection metrics computed in plain PyTorch/NumPy."""

from yolali.eval.evaluator import EvalConfig, Evaluator
from yolali.eval.map import DetectionMetrics, MetricAccumulator, average_precision
from yolali.eval.matching import IOU_THRESHOLDS, match_predictions

__all__ = [
    "IOU_THRESHOLDS",
    "DetectionMetrics",
    "EvalConfig",
    "Evaluator",
    "MetricAccumulator",
    "average_precision",
    "match_predictions",
]
