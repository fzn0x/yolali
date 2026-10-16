"""Anchor points for anchor-free detection heads."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import Tensor


def make_anchors(
    feat_shapes: Sequence[tuple[int, int]],
    strides: Sequence[int],
    *,
    offset: float = 0.5,
    device: torch.device | str | None = None,
    dtype: torch.dtype = torch.float32,
) -> tuple[Tensor, Tensor]:
    """One anchor point per feature-map cell, in pixels.

    Args:
        feat_shapes: ``[(H_i, W_i), ...]`` per level, in stride order.
        strides: ``[s_i, ...]`` matching ``feat_shapes``.
        offset: cell-centre offset in cells; ``0.5`` puts anchors at centres.
        device: where to allocate the result.
        dtype: floating dtype of the result.

    Returns:
        ``anchors``: ``(A, 2)`` ``[x, y]`` pixels, levels concatenated, each
        level in row-major (``y`` outer, ``x`` inner) order.
        ``anchor_strides``: ``(A, 1)`` stride of each anchor's level.
    """
    if len(feat_shapes) != len(strides):
        raise ValueError("feat_shapes and strides must have the same length")
    points: list[Tensor] = []
    stride_cols: list[Tensor] = []
    for (h, w), s in zip(feat_shapes, strides, strict=True):
        ys = (torch.arange(h, device=device, dtype=dtype) + offset) * s
        xs = (torch.arange(w, device=device, dtype=dtype) + offset) * s
        grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
        points.append(torch.stack((grid_x, grid_y), dim=-1).reshape(-1, 2))
        stride_cols.append(torch.full((h * w, 1), float(s), device=device, dtype=dtype))
    return torch.cat(points), torch.cat(stride_cols)
