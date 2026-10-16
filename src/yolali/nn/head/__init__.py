"""Detection heads: neck features -> raw box and class logits per anchor."""

from yolali.nn.head.detect import DetectHead
from yolali.nn.head.dual import DualDetectHead
from yolali.types import DualRawPrediction, RawPrediction

__all__ = ["DetectHead", "DualDetectHead", "DualRawPrediction", "RawPrediction"]
