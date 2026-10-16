"""Random affine / perspective warp with box filtering."""

from __future__ import annotations

import math

import cv2
import numpy as np

from yolali.data.sample import Sample
from yolali.data.transforms.letterbox import GRAY


def filter_boxes(
    before: np.ndarray,
    after: np.ndarray,
    *,
    min_size: float = 2.0,
    area_ratio: float = 0.1,
    max_aspect: float = 100.0,
) -> np.ndarray:
    """Keep boxes that survived a warp: ``(N,) bool``.

    A box is dropped when either side is under ``min_size`` px, its area fell
    below ``area_ratio`` of the original, or it became extremely elongated.
    """
    w1, h1 = before[:, 2] - before[:, 0], before[:, 3] - before[:, 1]
    w2, h2 = after[:, 2] - after[:, 0], after[:, 3] - after[:, 1]
    aspect = np.maximum(w2 / (h2 + 1e-16), h2 / (w2 + 1e-16))
    return (
        (w2 > min_size)
        & (h2 > min_size)
        & (w2 * h2 / (w1 * h1 + 1e-16) > area_ratio)
        & (aspect < max_aspect)
    )


class RandomPerspective:
    """Random rotation, translation, scale, shear and perspective.

    Args:
        degrees: rotation range in degrees (``±``).
        translate: translation as a fraction of the output size (``±``).
        scale: scale gain range ``1 ± scale``.
        shear: shear range in degrees (``±``).
        perspective: perspective coefficient range (``±``); ``0`` keeps the warp affine.
        border: ``(dy, dx)`` added to the output size. Negative values crop,
            which is how a ``2sx2s`` mosaic canvas becomes ``sxs``.
    """

    def __init__(
        self,
        *,
        degrees: float = 0.0,
        translate: float = 0.1,
        scale: float = 0.5,
        shear: float = 0.0,
        perspective: float = 0.0,
        border: tuple[int, int] = (0, 0),
    ) -> None:
        self.degrees = degrees
        self.translate = translate
        self.scale = scale
        self.shear = shear
        self.perspective = perspective
        self.border = border

    def matrix(
        self, hw: tuple[int, int], rng: np.random.Generator
    ) -> tuple[np.ndarray, tuple[int, int]]:
        """Sample the ``3x3`` warp and the output ``(H, W)``."""
        h, w = hw
        out_h, out_w = h + self.border[0] * 2, w + self.border[1] * 2

        center = np.eye(3)
        center[0, 2], center[1, 2] = -w / 2, -h / 2

        persp = np.eye(3)
        persp[2, 0] = rng.uniform(-self.perspective, self.perspective)
        persp[2, 1] = rng.uniform(-self.perspective, self.perspective)

        rot = np.eye(3)
        angle = rng.uniform(-self.degrees, self.degrees)
        gain = rng.uniform(1 - self.scale, 1 + self.scale)
        rot[:2] = cv2.getRotationMatrix2D(angle=angle, center=(0.0, 0.0), scale=gain)

        shear = np.eye(3)
        shear[0, 1] = math.tan(rng.uniform(-self.shear, self.shear) * math.pi / 180)
        shear[1, 0] = math.tan(rng.uniform(-self.shear, self.shear) * math.pi / 180)

        trans = np.eye(3)
        trans[0, 2] = rng.uniform(0.5 - self.translate, 0.5 + self.translate) * out_w
        trans[1, 2] = rng.uniform(0.5 - self.translate, 0.5 + self.translate) * out_h

        return trans @ shear @ rot @ persp @ center, (out_h, out_w)

    def __call__(self, sample: Sample, rng: np.random.Generator) -> Sample:
        """Warp image and boxes; boxes become the bounding box of their warped corners."""
        m, (out_h, out_w) = self.matrix(sample.hw, rng)
        is_identity = np.allclose(m, np.eye(3)) and (out_h, out_w) == sample.hw
        if is_identity:
            return sample
        if self.perspective:
            image = cv2.warpPerspective(sample.image, m, dsize=(out_w, out_h), borderValue=GRAY)
        else:
            image = cv2.warpAffine(sample.image, m[:2], dsize=(out_w, out_h), borderValue=GRAY)

        boxes = sample.boxes
        if len(boxes) == 0:
            return sample.with_(image=image)
        n = len(boxes)
        corners = np.ones((n * 4, 3))
        corners[:, :2] = boxes[:, [0, 1, 2, 3, 0, 3, 2, 1]].reshape(
            n * 4, 2
        )  # x1y1, x2y2, x1y2, x2y1
        warped = corners @ m.T
        warped = warped[:, :2] / warped[:, 2:3] if self.perspective else warped[:, :2]
        warped = warped.reshape(n, 8)
        xs, ys = warped[:, [0, 2, 4, 6]], warped[:, [1, 3, 5, 7]]
        new = np.stack((xs.min(1), ys.min(1), xs.max(1), ys.max(1)), 1)
        new[:, [0, 2]] = new[:, [0, 2]].clip(0, out_w)
        new[:, [1, 3]] = new[:, [1, 3]].clip(0, out_h)
        keep = filter_boxes(boxes, new)
        return sample.with_(
            image=image, boxes=new[keep].astype(np.float32), classes=sample.classes[keep]
        )
