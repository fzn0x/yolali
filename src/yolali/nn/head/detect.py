"""Single detection head with either DFL or direct box regression.

Layout follows YOLOv9's ``DDetect`` (Wang, Yeh, Liao 2024, arXiv:2402.13616)
so official checkpoints convert directly: a box branch and a class branch per
level, each two ``3x3`` convs then a ``1x1``. The box branch uses four groups
in its second and third convs when regressing DFL bins.

The head emits *raw* tensors. Decoding lives in :mod:`yolali.ops.decode`.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch
from torch import Tensor, nn

from yolali.nn.blocks import Conv, make_divisible
from yolali.ops.anchors import make_anchors
from yolali.types import RawPrediction


class DetectHead(nn.Module):
    """Per-level box and class branches.

    Args:
        in_channels: neck output channels ``(C3, C4, C5)``.
        num_classes: ``nc``.
        reg_max: DFL bins per side; ``0`` for direct ``ltrb`` regression.
        strides: stride of each level, used for anchors and bias init.
    """

    def __init__(
        self,
        in_channels: Sequence[int],
        num_classes: int,
        reg_max: int = 16,
        strides: Sequence[int] = (8, 16, 32),
    ) -> None:
        super().__init__()
        if len(in_channels) != len(strides):
            raise ValueError("one stride per input level is required")
        self.num_classes = num_classes
        self.reg_max = reg_max
        self.strides = tuple(strides)
        box_out = 4 * reg_max if reg_max > 0 else 4
        groups = 4 if reg_max > 0 else 1
        c_box = make_divisible(max(in_channels[0] // 4, box_out, 16), 4)
        c_cls = max(in_channels[0], min(num_classes * 2, 128))

        self.box = nn.ModuleList(
            nn.Sequential(
                Conv(c, c_box, 3),
                Conv(c_box, c_box, 3, g=groups),
                nn.Conv2d(c_box, box_out, 1, groups=groups),
            )
            for c in in_channels
        )
        self.cls = nn.ModuleList(
            nn.Sequential(
                Conv(c, c_cls, 3), Conv(c_cls, c_cls, 3), nn.Conv2d(c_cls, num_classes, 1)
            )
            for c in in_channels
        )
        self.init_biases()

    @property
    def box_channels(self) -> int:
        """Channels of ``box_raw``: ``4 * reg_max`` or ``4``."""
        return 4 * self.reg_max if self.reg_max > 0 else 4

    def init_biases(self) -> None:
        """Prior-aware bias init: boxes start at one stride, classes at ~5 objects/image."""
        for box, cls, s in zip(self.box, self.cls, self.strides, strict=True):
            assert isinstance(box, nn.Sequential) and isinstance(cls, nn.Sequential)
            box_final, cls_final = box[-1], cls[-1]
            assert isinstance(box_final, nn.Conv2d) and isinstance(cls_final, nn.Conv2d)
            assert box_final.bias is not None and cls_final.bias is not None
            with torch.no_grad():
                box_final.bias.fill_(1.0)
                cls_final.bias.fill_(math.log(5 / self.num_classes / (640 / s) ** 2))

    def forward(self, feats: list[Tensor]) -> RawPrediction:
        """``[(B, C_i, H_i, W_i)]`` -> flattened raw predictions with anchors."""
        boxes: list[Tensor] = []
        classes: list[Tensor] = []
        shapes: list[tuple[int, int]] = []
        for x, box, cls in zip(feats, self.box, self.cls, strict=True):
            b, _, h, w = x.shape
            boxes.append(box(x).view(b, self.box_channels, h * w))
            classes.append(cls(x).view(b, self.num_classes, h * w))
            shapes.append((int(h), int(w)))
        anchors, strides = make_anchors(
            shapes, self.strides, device=feats[0].device, dtype=feats[0].dtype
        )
        return RawPrediction(
            box_raw=torch.cat(boxes, 2).permute(0, 2, 1),
            cls_logits=torch.cat(classes, 2).permute(0, 2, 1),
            anchors=anchors,
            strides=strides,
        )
