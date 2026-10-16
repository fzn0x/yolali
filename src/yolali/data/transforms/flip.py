"""Horizontal / vertical flips."""

from __future__ import annotations

import numpy as np

from yolali.data.sample import Sample


class RandomFlip:
    """Flip with probability ``p``; ``horizontal=False`` flips vertically."""

    def __init__(self, p: float = 0.5, horizontal: bool = True) -> None:
        self.p = p
        self.horizontal = horizontal

    def __call__(self, sample: Sample, rng: np.random.Generator) -> Sample:
        """Mirror the image and the matching box coordinates."""
        if rng.random() >= self.p:
            return sample
        h, w = sample.hw
        boxes = sample.boxes.copy()
        if self.horizontal:
            image = np.ascontiguousarray(sample.image[:, ::-1])
            boxes[:, [0, 2]] = w - boxes[:, [2, 0]]
        else:
            image = np.ascontiguousarray(sample.image[::-1])
            boxes[:, [1, 3]] = h - boxes[:, [3, 1]]
        return sample.with_(image=image, boxes=boxes)
