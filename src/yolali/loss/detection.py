"""The loss for one detection head: assign, then classify and regress."""

from __future__ import annotations

from dataclasses import dataclass

from torch import Tensor, nn

from yolali.assign.tal import TaskAlignedAssigner
from yolali.loss.box import dfl_loss, iou_loss
from yolali.loss.cls import bce_loss, varifocal_loss
from yolali.loss.config import LossConfig
from yolali.ops.boxes import xyxy_to_ltrb
from yolali.ops.decode import decode_boxes
from yolali.types import RawPrediction, Targets


@dataclass(frozen=True)
class LossOutput:
    """Scalar loss terms for one head. ``total`` is what you backpropagate."""

    total: Tensor
    box: Tensor
    cls: Tensor
    dfl: Tensor
    num_fg: int

    def as_dict(self, prefix: str = "") -> dict[str, float]:
        """Detached floats for logging."""
        return {
            f"{prefix}total": float(self.total.detach()),
            f"{prefix}box": float(self.box.detach()),
            f"{prefix}cls": float(self.cls.detach()),
            f"{prefix}dfl": float(self.dfl.detach()),
            f"{prefix}num_fg": float(self.num_fg),
        }


class DetectionLoss(nn.Module):
    """Loss for a head that outputs raw box logits and class logits.

    The model is not involved: pass the head's raw outputs, the anchors and
    strides the head was evaluated on, and the targets.

    Args:
        cfg: hyper-parameters.
        num_classes: ``nc``.
        assigner: defaults to a :class:`TaskAlignedAssigner` built from ``cfg``.
            Pass a :class:`~yolali.assign.OneToOneAssigner` for the o2o branch.

    Normalisation follows the YOLOv8/v9 convention: every term is divided by
    the total soft-target mass (clamped to at least 1) and the final sum is
    multiplied by the batch size, so the default weights and learning rate
    transfer across batch sizes.
    """

    def __init__(
        self, cfg: LossConfig, num_classes: int, assigner: nn.Module | None = None
    ) -> None:
        super().__init__()
        self.cfg = cfg
        self.num_classes = num_classes
        self.assigner = assigner or TaskAlignedAssigner(cfg.tal_topk, cfg.tal_alpha, cfg.tal_beta)

    def forward(self, pred: RawPrediction, targets: Targets) -> LossOutput:
        """Loss of a head's raw output against padded ground truth in pixels."""
        return self.compute(pred.box_raw, pred.cls_logits, pred.anchors, pred.strides, targets)

    def compute(
        self,
        box_raw: Tensor,
        cls_logits: Tensor,
        anchors: Tensor,
        strides: Tensor,
        targets: Targets,
    ) -> LossOutput:
        """Compute the loss from bare tensors.

        Args:
            box_raw: ``(B, A, 4 * reg_max)`` DFL logits, or ``(B, A, 4)`` when
                ``cfg.reg_max == 0``.
            cls_logits: ``(B, A, nc)`` pre-sigmoid.
            anchors: ``(A, 2)`` pixel centres.
            strides: ``(A, 1)`` per-anchor stride.
            targets: padded ground truth in pixels.
        """
        cfg = self.cfg
        batch = box_raw.shape[0]
        pred_boxes = decode_boxes(box_raw, anchors, strides, cfg.reg_max)  # (B, A, 4) px

        assigned = self.assigner(
            cls_logits.detach().sigmoid(), pred_boxes.detach(), anchors, targets
        )
        target_mass = assigned.scores.sum().clamp(min=1.0)
        fg = assigned.fg_mask  # (B, A)
        num_fg = int(fg.sum())

        if cfg.cls_loss == "vfl":
            cls = varifocal_loss(cls_logits, assigned.scores, cfg.vfl_alpha, cfg.vfl_gamma)
        else:
            cls = bce_loss(cls_logits, assigned.scores)
        cls = cls / target_mass

        zero = box_raw.sum() * 0.0  # keeps the graph connected when there are no positives
        box = zero
        dfl = zero
        if num_fg > 0:
            weight = assigned.scores.sum(-1)[fg]  # (N,)
            box = iou_loss(pred_boxes[fg], assigned.boxes[fg], weight, cfg.iou_kind) / target_mass
            if cfg.reg_max > 0:
                anchors_fg = anchors.unsqueeze(0).expand(batch, -1, -1)[fg]  # (N, 2)
                strides_fg = strides.unsqueeze(0).expand(batch, -1, -1)[fg]  # (N, 1)
                target_ltrb = xyxy_to_ltrb(anchors_fg, assigned.boxes[fg]) / strides_fg
                dfl = dfl_loss(box_raw[fg], target_ltrb, weight, cfg.reg_max) / target_mass

        total = (cfg.box_weight * box + cfg.cls_weight * cls + cfg.dfl_weight * dfl) * batch
        return LossOutput(total, box, cls, dfl, num_fg)


def loss_terms_to_dict(*outputs: tuple[str, LossOutput]) -> dict[str, float]:
    """Merge several named :class:`LossOutput` objects into one flat log dict."""
    merged: dict[str, float] = {}
    for name, out in outputs:
        merged.update(out.as_dict(prefix=f"{name}/"))
    return merged
