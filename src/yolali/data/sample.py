"""The unit that flows through the data pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field, replace

import numpy as np
import numpy.typing as npt

UInt8Image = npt.NDArray[np.uint8]  # (H, W, 3) RGB
Boxes = npt.NDArray[np.float32]  # (N, 4) xyxy pixels
Classes = npt.NDArray[np.int64]  # (N,)


@dataclass
class Sample:
    """One image with its boxes, plus bookkeeping to undo the letterbox.

    Attributes:
        image: ``(H, W, 3)`` uint8 RGB.
        boxes: ``(N, 4)`` float32 ``xyxy`` in pixels of ``image``.
        classes: ``(N,)`` int64.
        path: where the image came from (for logging and evaluation).
        orig_hw: height and width before any transform.
        scale: resize factor applied by :class:`~yolali.data.transforms.Letterbox`.
        pad: ``(left, top)`` padding in pixels added by the letterbox.
    """

    image: UInt8Image
    boxes: Boxes
    classes: Classes
    path: str = ""
    orig_hw: tuple[int, int] = field(default=(0, 0))
    scale: float = 1.0
    pad: tuple[float, float] = (0.0, 0.0)

    def __post_init__(self) -> None:
        if self.image.ndim != 3 or self.image.shape[2] != 3:
            raise ValueError(f"image must be (H, W, 3), got {self.image.shape}")
        self.boxes = np.asarray(self.boxes, dtype=np.float32).reshape(-1, 4)
        self.classes = np.asarray(self.classes, dtype=np.int64).reshape(-1)
        if len(self.boxes) != len(self.classes):
            raise ValueError("boxes and classes must have the same length")
        if self.orig_hw == (0, 0):
            self.orig_hw = (int(self.image.shape[0]), int(self.image.shape[1]))

    @property
    def hw(self) -> tuple[int, int]:
        """Current ``(height, width)``."""
        return int(self.image.shape[0]), int(self.image.shape[1])

    def with_(self, **changes: object) -> Sample:
        """Shallow copy with some fields replaced (``dataclasses.replace``)."""
        return replace(self, **changes)  # pyright: ignore[reportArgumentType] - replace accepts field kwargs
