"""Turn raw head outputs into ``xyxy`` pixel boxes.

Two regression parameterisations are supported, selected by ``reg_max``:

* ``reg_max > 0``: Distribution Focal Loss (DFL) bins, as in YOLOv9's
  ``DDetect`` (Wang et al. 2024, arXiv:2402.13616). Each side distance is the
  expectation of a softmax over ``reg_max`` integer bins, in stride units.
* ``reg_max == 0``: direct regression of the four distances (YOLO26 style).
  Non-negativity comes from ``softplus``; see ``docs/design.md`` D2.

The same functions serve training, inference and export so the three can
never disagree.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F  # noqa: N812 - torch convention
from torch import Tensor

from yolali.ops.boxes import ltrb_to_xyxy


def dfl_expectation(box_raw: Tensor, reg_max: int) -> Tensor:
    """DFL logits -> expected distances in stride units.

    Args:
        box_raw: ``(B, A, 4 * reg_max)`` logits, side-major: the first
            ``reg_max`` channels are the bins of ``left``, then ``top``, ...
        reg_max: number of bins per side.

    Returns:
        ``(B, A, 4)`` ``[l, t, r, b]`` in ``[0, reg_max - 1]`` stride units.
    """
    b, a, _ = box_raw.shape
    probs = box_raw.view(b, a, 4, reg_max).softmax(dim=-1)
    bins = torch.arange(reg_max, device=box_raw.device, dtype=probs.dtype)
    return (probs * bins).sum(-1)


def direct_distances(box_raw: Tensor) -> Tensor:
    """Raw direct-regression output -> non-negative distances in stride units.

    ``(B, A, 4)`` -> ``(B, A, 4)`` via ``softplus``.
    """
    return F.softplus(box_raw)


def decode_boxes(box_raw: Tensor, anchors: Tensor, strides: Tensor, reg_max: int) -> Tensor:
    """Raw head output -> ``(B, A, 4)`` ``xyxy`` boxes in pixels.

    Args:
        box_raw: ``(B, A, 4 * reg_max)`` if ``reg_max > 0`` else ``(B, A, 4)``.
        anchors: ``(A, 2)`` pixel anchor centres.
        strides: ``(A, 1)`` per-anchor stride.
        reg_max: DFL bins, or ``0`` for direct regression.
    """
    ltrb = dfl_expectation(box_raw, reg_max) if reg_max > 0 else direct_distances(box_raw)
    return ltrb_to_xyxy(anchors, ltrb * strides)
