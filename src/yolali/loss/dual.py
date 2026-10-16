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
from yolali.types import Targets


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

    def forward(
        self,
        o2m_outputs: tuple[Tensor, Tensor],
        o2o_outputs: tuple[Tensor, Tensor],
        anchors: Tensor,
        strides: Tensor,
        targets: Targets,
    ) -> DualLossOutput:
        """Each ``*_outputs`` is ``(box_raw, cls_logits)`` from that branch."""
        o2m = self.o2m(o2m_outputs[0], o2m_outputs[1], anchors, strides, targets)
        o2o = self.o2o(o2o_outputs[0], o2o_outputs[1], anchors, strides, targets)
        return DualLossOutput(o2m.total + self.cfg.o2o_weight * o2o.total, o2m, o2o)
