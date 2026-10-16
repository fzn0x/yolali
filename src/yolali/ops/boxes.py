"""Box format conversions and the IoU family.

All boxes are ``xyxy`` in pixels unless the function name says otherwise.
Shapes are written ``(..., 4)``: any leading batch dimensions broadcast.

References:
    GIoU: Rezatofighi et al., "Generalized Intersection over Union", CVPR 2019.
    CIoU: Zheng et al., "Distance-IoU Loss", AAAI 2020.
"""

from __future__ import annotations

import math
from typing import Literal

import torch
from torch import Tensor

IoUKind = Literal["iou", "giou", "ciou"]

_EPS = 1e-7


def unique_values(x: Tensor) -> list[int]:
    """Sorted distinct integer values of a 1-D tensor, as Python ints."""
    values: Tensor = torch.unique(x)  # pyright: ignore[reportUnknownVariableType] - overloaded stub
    return [int(v) for v in values.tolist()]  # pyright: ignore[reportUnknownVariableType] - as above


def xyxy_to_cxcywh(boxes: Tensor) -> Tensor:
    """``(..., 4) [x1, y1, x2, y2]`` -> ``(..., 4) [cx, cy, w, h]``."""
    x1, y1, x2, y2 = boxes.unbind(-1)
    return torch.stack(((x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1), dim=-1)


def cxcywh_to_xyxy(boxes: Tensor) -> Tensor:
    """``(..., 4) [cx, cy, w, h]`` -> ``(..., 4) [x1, y1, x2, y2]``."""
    cx, cy, w, h = boxes.unbind(-1)
    return torch.stack((cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2), dim=-1)


def ltrb_to_xyxy(anchors: Tensor, ltrb: Tensor) -> Tensor:
    """Distances from an anchor point to the four box sides -> ``xyxy``.

    Args:
        anchors: ``(..., 2)`` anchor centres ``[ax, ay]`` in pixels.
        ltrb: ``(..., 4)`` distances ``[left, top, right, bottom]`` in pixels.

    Returns:
        ``(..., 4)`` boxes ``[ax - l, ay - t, ax + r, ay + b]``.
    """
    ax, ay = anchors.unbind(-1)
    left, top, right, bottom = ltrb.unbind(-1)
    return torch.stack((ax - left, ay - top, ax + right, ay + bottom), dim=-1)


def xyxy_to_ltrb(anchors: Tensor, boxes: Tensor) -> Tensor:
    """Inverse of :func:`ltrb_to_xyxy`; distances in pixels, may be negative."""
    ax, ay = anchors.unbind(-1)
    x1, y1, x2, y2 = boxes.unbind(-1)
    return torch.stack((ax - x1, ay - y1, x2 - ax, y2 - ay), dim=-1)


def broadcast_iou(a: Tensor, b: Tensor, kind: IoUKind) -> Tensor:
    """IoU variant for broadcastable ``(..., 4)`` box tensors -> ``(...)``.

    Shared by :func:`box_iou` (all pairs) and :func:`elementwise_iou` (1:1) so
    there is exactly one place where the formulas live.
    """
    ax1, ay1, ax2, ay2 = a.unbind(-1)
    bx1, by1, bx2, by2 = b.unbind(-1)

    inter_w = (torch.minimum(ax2, bx2) - torch.maximum(ax1, bx1)).clamp_(min=0)
    inter_h = (torch.minimum(ay2, by2) - torch.maximum(ay1, by1)).clamp_(min=0)
    inter = inter_w * inter_h

    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    union = area_a + area_b - inter + _EPS
    iou = inter / union
    if kind == "iou":
        return iou

    # Smallest enclosing box of the pair.
    enc_w = torch.maximum(ax2, bx2) - torch.minimum(ax1, bx1)
    enc_h = torch.maximum(ay2, by2) - torch.minimum(ay1, by1)

    if kind == "giou":
        enc_area = enc_w * enc_h + _EPS
        return iou - (enc_area - union) / enc_area

    # CIoU: subtract normalised centre distance and an aspect-ratio penalty.
    c2 = enc_w.pow(2) + enc_h.pow(2) + _EPS
    rho2 = ((ax1 + ax2) - (bx1 + bx2)).pow(2) / 4 + ((ay1 + ay2) - (by1 + by2)).pow(2) / 4
    wa, ha = ax2 - ax1, ay2 - ay1
    wb, hb = bx2 - bx1, by2 - by1
    v = (4 / math.pi**2) * (torch.atan(wb / (hb + _EPS)) - torch.atan(wa / (ha + _EPS))).pow(2)
    with torch.no_grad():
        alpha = v / (v - iou + (1 + _EPS))
    return iou - (rho2 / c2 + v * alpha)


def box_iou(a: Tensor, b: Tensor, kind: IoUKind = "iou") -> Tensor:
    """All-pairs IoU.

    Args:
        a: ``(N, 4)`` xyxy.
        b: ``(M, 4)`` xyxy.
        kind: ``"iou"``, ``"giou"`` or ``"ciou"``.

    Returns:
        ``(N, M)`` matrix; entry ``[i, j]`` is the IoU of ``a[i]`` and ``b[j]``.
    """
    return broadcast_iou(a.unsqueeze(-2), b.unsqueeze(-3), kind)


def elementwise_iou(a: Tensor, b: Tensor, kind: IoUKind = "iou") -> Tensor:
    """Row-wise IoU of two equally shaped ``(..., 4)`` tensors -> ``(...)``."""
    return broadcast_iou(a, b, kind)
