"""Transform protocol and composition."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

import numpy as np

from yolali.data.sample import Sample


class Transform(Protocol):
    """Anything callable on a :class:`Sample` that returns a :class:`Sample`."""

    def __call__(self, sample: Sample, rng: np.random.Generator) -> Sample:
        """Apply the transform, drawing randomness only from ``rng``."""
        ...


class Compose:
    """Apply transforms in order."""

    def __init__(self, transforms: Sequence[Transform]) -> None:
        self.transforms = list(transforms)

    def __call__(self, sample: Sample, rng: np.random.Generator) -> Sample:
        """Run every transform with the same generator."""
        for t in self.transforms:
            sample = t(sample, rng)
        return sample

    def __repr__(self) -> str:
        inner = ", ".join(type(t).__name__ for t in self.transforms)
        return f"Compose([{inner}])"
