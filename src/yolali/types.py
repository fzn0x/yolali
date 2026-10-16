"""Small shared containers that cross component boundaries.

Everything here is a frozen dataclass of tensors with the shape convention
from ``docs/design.md`` section 3: boxes are ``xyxy`` in pixels.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass(frozen=True)
class Targets:
    """Padded ground-truth boxes for one batch.

    Attributes:
        boxes: ``(B, M, 4)`` float ``xyxy`` in pixels. Padding rows are zeros.
        classes: ``(B, M)`` int64 class indices. Padding rows are zeros.
        mask: ``(B, M)`` bool, ``True`` for real boxes.
    """

    boxes: Tensor
    classes: Tensor
    mask: Tensor

    def __post_init__(self) -> None:
        if self.boxes.ndim != 3 or self.boxes.shape[-1] != 4:
            raise ValueError(f"boxes must be (B, M, 4), got {tuple(self.boxes.shape)}")
        if self.classes.shape != self.boxes.shape[:2]:
            raise ValueError("classes must be (B, M) matching boxes")
        if self.mask.shape != self.boxes.shape[:2]:
            raise ValueError("mask must be (B, M) matching boxes")

    @property
    def batch_size(self) -> int:
        """Number of images ``B``."""
        return int(self.boxes.shape[0])

    @property
    def max_boxes(self) -> int:
        """Padded box count ``M``."""
        return int(self.boxes.shape[1])

    @classmethod
    def from_list(cls, boxes: Sequence[Tensor], classes: Sequence[Tensor]) -> Targets:
        """Pad per-image ``(N_i, 4)`` boxes and ``(N_i,)`` classes to ``(B, M, ...)``.

        An image with no boxes contributes an all-``False`` mask row.
        ``M`` is ``max(N_i)``, or ``1`` when every image is empty so that
        downstream code never sees a zero-width dimension.
        """
        if len(boxes) != len(classes):
            raise ValueError("boxes and classes must have the same length")
        batch = len(boxes)
        max_n = max((int(b.shape[0]) for b in boxes), default=0)
        max_n = max(max_n, 1)
        ref = boxes[0] if batch else torch.zeros(0, 4)
        out_boxes = torch.zeros(batch, max_n, 4, dtype=ref.dtype, device=ref.device)
        out_cls = torch.zeros(batch, max_n, dtype=torch.int64, device=ref.device)
        out_mask = torch.zeros(batch, max_n, dtype=torch.bool, device=ref.device)
        for i, (b, c) in enumerate(zip(boxes, classes, strict=True)):
            n = int(b.shape[0])
            if n == 0:
                continue
            out_boxes[i, :n] = b
            out_cls[i, :n] = c.to(torch.int64)
            out_mask[i, :n] = True
        return cls(out_boxes, out_cls, out_mask)

    def to(self, device: torch.device | str) -> Targets:
        """Move every tensor to ``device``."""
        return Targets(self.boxes.to(device), self.classes.to(device), self.mask.to(device))


@dataclass(frozen=True)
class RawPrediction:
    """Detection-head output for one batch, all levels flattened in stride order.

    Attributes:
        box_raw: ``(B, A, 4 * reg_max)`` DFL logits or ``(B, A, 4)`` direct.
        cls_logits: ``(B, A, nc)`` pre-sigmoid.
        anchors: ``(A, 2)`` anchor centres in pixels.
        strides: ``(A, 1)`` stride of each anchor.
    """

    box_raw: Tensor
    cls_logits: Tensor
    anchors: Tensor
    strides: Tensor


@dataclass(frozen=True)
class DualRawPrediction:
    """Training-time output of a dual head: one-to-many and one-to-one branches."""

    o2m: RawPrediction
    o2o: RawPrediction
