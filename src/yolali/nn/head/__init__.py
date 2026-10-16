"""Detection heads: neck features -> raw box and class logits per anchor."""

from yolali.nn.head.detect import DetectHead, RawPrediction

__all__ = ["DetectHead", "RawPrediction"]
