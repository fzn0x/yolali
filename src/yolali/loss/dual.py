"""Loss for the dual (one-to-many + one-to-one) head used for NMS-free training.

After YOLOv10's consistent dual assignment (arXiv:2405.14458): the two
branches share backbone and neck, see the same targets, and use the same
alignment metric; only the number of anchors per box differs.
"""

from __future__ import annotations

from dataclasses import dataclass

from torch import Tensor, nn

from yolali.assign.o2o import OneToOneAssigner
from yolali.assign.tal import TaskAlignedAssigner
from yolali.loss.config import LossConfig
from yolali.loss.detection import DetectionLoss, LossOutput, loss_terms_to_dict
from yolali.types import DualRawPrediction, Targets


@dataclass(frozen=True)
class DualLossOutput:
    """Both branches' terms plus the combined scalar to backpropagate."""

    total: Tensor
    o2m: LossOutput
    o2o: LossOutput

    def as_dict(self) -> dict[str, float]:
        """Detached floats for logging, prefixed ``o2m/`` and ``o2o/``."""
        d = loss_terms_to_dict(("o2m", self.o2m), ("o2o", self.o2o))
        d["total"] = float(self.total.detach())
        return d


class DualLoss(nn.Module):
    """``o2m_loss + o2o_weight * o2o_loss`` over two heads' raw outputs."""

    def __init__(self, cfg: LossConfig, num_classes: int) -> None:
        super().__init__()
        self.cfg = cfg
        self.o2m = DetectionLoss(
            cfg, num_classes, TaskAlignedAssigner(cfg.tal_topk, cfg.tal_alpha, cfg.tal_beta)
        )
        self.o2o = DetectionLoss(cfg, num_classes, OneToOneAssigner(cfg.tal_alpha, cfg.tal_beta))

    def forward(self, pred: DualRawPrediction, targets: Targets) -> DualLossOutput:
        """Both branches see the same targets; only their assigners differ."""
        o2m = self.o2m(pred.o2m, targets)
        o2o = self.o2o(pred.o2o, targets)
        return DualLossOutput(o2m.total + self.cfg.o2o_weight * o2o.total, o2m, o2o)
