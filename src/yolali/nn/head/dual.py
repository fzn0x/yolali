"""Dual head for NMS-free detection.

After YOLOv10 (Wang et al. 2024, arXiv:2405.14458): two detection heads share
the backbone and neck. The *one-to-many* head is trained with the usual
task-aligned assignment (rich supervision); the *one-to-one* head is trained
with the same alignment metric but a single positive per object. At
inference only the one-to-one head runs, and its top-scoring anchors are the
final detections with no NMS.

The one-to-one branch sees *detached* neck features, so the backbone and
neck are shaped by the one-to-many supervision alone; the one-to-one head
only learns to pick its single anchor on top of them (``docs/design.md`` D3).
"""

from __future__ import annotations

from collections.abc import Sequence

from torch import Tensor, nn

from yolali.nn.head.detect import DetectHead
from yolali.types import DualRawPrediction, RawPrediction


class DualDetectHead(nn.Module):
    """Two :class:`DetectHead` branches on the same features.

    Args:
        in_channels: neck output channels ``(C3, C4, C5)``.
        num_classes: ``nc``.
        reg_max: DFL bins per side; ``0`` for direct regression (the default
            NMS-free configuration uses ``0``).
        strides: stride of each level.
        detach_o2o: feed the one-to-one branch detached features.
    """

    def __init__(
        self,
        in_channels: Sequence[int],
        num_classes: int,
        reg_max: int = 0,
        strides: Sequence[int] = (8, 16, 32),
        *,
        detach_o2o: bool = True,
    ) -> None:
        super().__init__()
        self.o2m = DetectHead(in_channels, num_classes, reg_max, strides)
        self.o2o = DetectHead(in_channels, num_classes, reg_max, strides)
        self.detach_o2o = detach_o2o
        self.num_classes = num_classes
        self.reg_max = reg_max

    def forward(self, feats: list[Tensor]) -> DualRawPrediction | RawPrediction:
        """Both branches while training; only the one-to-one branch in ``eval()``."""
        if not self.training:
            return self.o2o(feats)
        o2o_feats = [f.detach() for f in feats] if self.detach_o2o else feats
        return DualRawPrediction(o2m=self.o2m(feats), o2o=self.o2o(o2o_feats))
