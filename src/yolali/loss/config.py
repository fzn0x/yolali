"""Loss hyper-parameters as one frozen dataclass."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from yolali.ops.boxes import IoUKind

ClsLossKind = Literal["bce", "vfl"]


@dataclass(frozen=True)
class LossConfig:
    """Everything :class:`~yolali.loss.DetectionLoss` needs to know.

    Attributes:
        box_weight: multiplier on the IoU loss term.
        cls_weight: multiplier on the classification term.
        dfl_weight: multiplier on the DFL term; unused when ``reg_max == 0``.
        iou_kind: ``"ciou"`` or ``"giou"`` for the box term.
        cls_loss: ``"bce"`` (binary cross-entropy on soft targets) or ``"vfl"``
            (varifocal loss).
        reg_max: DFL bins per side; ``0`` selects direct ``ltrb`` regression.
        tal_topk: candidates per ground truth in the one-to-many assigner.
        tal_alpha: score exponent in the alignment metric.
        tal_beta: IoU exponent in the alignment metric.
        o2o_weight: weight of the one-to-one branch inside :class:`DualLoss`.
        vfl_alpha: varifocal negative-sample weight.
        vfl_gamma: varifocal focusing exponent.
    """

    box_weight: float = 7.5
    cls_weight: float = 0.5
    dfl_weight: float = 1.5
    iou_kind: IoUKind = "ciou"
    cls_loss: ClsLossKind = "bce"
    reg_max: int = 16
    tal_topk: int = 10
    tal_alpha: float = 0.5
    tal_beta: float = 6.0
    o2o_weight: float = 1.0
    vfl_alpha: float = 0.75
    vfl_gamma: float = 2.0

    def __post_init__(self) -> None:
        if self.reg_max < 0:
            raise ValueError("reg_max must be >= 0 (0 means direct regression)")
        if self.tal_topk < 1:
            raise ValueError("tal_topk must be >= 1")
        if self.iou_kind == "iou":
            raise ValueError("plain IoU has no gradient for disjoint boxes; use ciou or giou")
