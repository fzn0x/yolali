"""Detection losses, decoupled from any model."""

from yolali.loss.box import dfl_loss, iou_loss
from yolali.loss.cls import bce_loss, varifocal_loss
from yolali.loss.config import LossConfig
from yolali.loss.detection import DetectionLoss, LossOutput
from yolali.loss.dual import DualLoss, DualLossOutput

__all__ = [
    "DetectionLoss",
    "DualLoss",
    "DualLossOutput",
    "LossConfig",
    "LossOutput",
    "bce_loss",
    "dfl_loss",
    "iou_loss",
    "varifocal_loss",
]
