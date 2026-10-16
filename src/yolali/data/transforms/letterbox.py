"""Resize with preserved aspect ratio, then pad to the target shape."""

from __future__ import annotations

import math

import cv2
import numpy as np
from torch import Tensor

from yolali.data.sample import Sample

GRAY = (114, 114, 114)


def letterbox_params(
    hw: tuple[int, int], target: tuple[int, int], *, scale_up: bool = True
) -> tuple[float, tuple[int, int], tuple[float, float]]:
    """``(scale, (new_h, new_w), (pad_left, pad_top))`` to fit ``hw`` into ``target``."""
    h, w = hw
    th, tw = target
    scale = min(th / h, tw / w)
    if not scale_up:
        scale = min(scale, 1.0)
    new_h, new_w = round(h * scale), round(w * scale)
    pad_left = (tw - new_w) / 2
    pad_top = (th - new_h) / 2
    return scale, (new_h, new_w), (pad_left, pad_top)


class Letterbox:
    """Fit the image inside ``size`` and pad the rest with gray.

    Args:
        size: output ``(H, W)``; an ``int`` means a square.
        stride: with ``auto=True``, pad only to the next multiple of ``stride``
            (minimal rectangle) instead of the full ``size``.
        auto: see ``stride``.
        scale_up: allow enlarging small images.
    """

    def __init__(
        self,
        size: int | tuple[int, int],
        *,
        stride: int = 32,
        auto: bool = False,
        scale_up: bool = True,
    ) -> None:
        self.size = (size, size) if isinstance(size, int) else size
        self.stride = stride
        self.auto = auto
        self.scale_up = scale_up

    def __call__(self, sample: Sample, rng: np.random.Generator) -> Sample:
        """Resize and pad; records ``scale`` and ``pad`` on the sample."""
        scale, (new_h, new_w), _ = letterbox_params(sample.hw, self.size, scale_up=self.scale_up)
        if self.auto:
            target = (
                math.ceil(new_h / self.stride) * self.stride,
                math.ceil(new_w / self.stride) * self.stride,
            )
        else:
            target = self.size
        pad_left = (target[1] - new_w) / 2
        pad_top = (target[0] - new_h) / 2

        image = sample.image
        if (new_h, new_w) != sample.hw:
            image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        top, bottom = round(pad_top - 0.1), round(pad_top + 0.1)
        left, right = round(pad_left - 0.1), round(pad_left + 0.1)
        image = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=GRAY)

        boxes = sample.boxes.copy()
        boxes[:, [0, 2]] = boxes[:, [0, 2]] * scale + left
        boxes[:, [1, 3]] = boxes[:, [1, 3]] * scale + top
        return sample.with_(
            image=image, boxes=boxes, scale=sample.scale * scale, pad=(float(left), float(top))
        )


def undo_letterbox(
    boxes: Tensor, scale: float, pad: tuple[float, float], orig_hw: tuple[int, int]
) -> Tensor:
    """Map ``(N, 4)`` letterboxed xyxy boxes back to original-image pixels, clipped."""
    out = boxes.clone()
    out[:, [0, 2]] = (out[:, [0, 2]] - pad[0]) / scale
    out[:, [1, 3]] = (out[:, [1, 3]] - pad[1]) / scale
    out[:, [0, 2]] = out[:, [0, 2]].clamp(0, orig_hw[1])
    out[:, [1, 3]] = out[:, [1, 3]].clamp(0, orig_hw[0])
    return out
