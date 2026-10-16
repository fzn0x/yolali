"""Colour jitter in HSV space."""

from __future__ import annotations

import cv2
import numpy as np

from yolali.data.sample import Sample


class RandomHSV:
    """Multiply hue, saturation and value by random gains in ``1 ± (h, s, v)``.

    Hue wraps modulo 180 (OpenCV's 8-bit hue range); S and V saturate at 255.
    """

    def __init__(self, h: float = 0.015, s: float = 0.7, v: float = 0.4) -> None:
        self.h, self.s, self.v = h, s, v

    def __call__(self, sample: Sample, rng: np.random.Generator) -> Sample:
        """Apply the jitter; boxes are untouched."""
        if self.h == 0 and self.s == 0 and self.v == 0:
            return sample
        gains = rng.uniform(-1, 1, 3) * np.array([self.h, self.s, self.v]) + 1
        hue, sat, val = cv2.split(cv2.cvtColor(sample.image, cv2.COLOR_RGB2HSV))
        x = np.arange(256, dtype=np.float32)
        lut_h = ((x * gains[0]) % 180).astype(np.uint8)
        lut_s = np.clip(x * gains[1], 0, 255).astype(np.uint8)
        lut_v = np.clip(x * gains[2], 0, 255).astype(np.uint8)
        hsv = cv2.merge((cv2.LUT(hue, lut_h), cv2.LUT(sat, lut_s), cv2.LUT(val, lut_v)))
        image = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
        return sample.with_(image=image.astype(np.uint8))
