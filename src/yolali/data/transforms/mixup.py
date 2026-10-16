"""MixUp: blend two samples and keep both label sets."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from yolali.data.sample import Sample

Sampler = Callable[[np.random.Generator], Sample]


class MixUp:
    """``image = r * a + (1 - r) * b`` with ``r ~ Beta(alpha, alpha)``.

    The partner must already have the same ``(H, W)``; the dataset's sampler
    is responsible for running the partner through the same geometric
    transforms first.
    """

    def __init__(self, sampler: Sampler, p: float = 0.0, alpha: float = 32.0) -> None:
        self.sampler = sampler
        self.p = p
        self.alpha = alpha

    def __call__(self, sample: Sample, rng: np.random.Generator) -> Sample:
        """Blend with a partner sample and concatenate labels."""
        if rng.random() >= self.p:
            return sample
        other = self.sampler(rng)
        if other.hw != sample.hw:
            raise ValueError(f"MixUp partner shape {other.hw} != {sample.hw}; letterbox both first")
        r = float(rng.beta(self.alpha, self.alpha))
        image = (
            sample.image.astype(np.float32) * r + other.image.astype(np.float32) * (1 - r)
        ).astype(np.uint8)
        return sample.with_(
            image=image,
            boxes=np.concatenate((sample.boxes, other.boxes)),
            classes=np.concatenate((sample.classes, other.classes)),
        )
