"""The label record every reader produces."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import numpy.typing as npt
from PIL import Image

IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff")


@dataclass
class LabelRecord:
    """Annotations of one image, boxes in ``xyxy`` pixels.

    Attributes:
        path: absolute image path.
        width: image width in pixels.
        height: image height in pixels.
        boxes: ``(N, 4)`` float32 ``xyxy`` pixels, clipped to the image.
        classes: ``(N,)`` int64 contiguous class indices.
        crowd: ``(N,)`` bool, ``True`` for COCO ``iscrowd`` regions (ignored
            by the loss, handled specially by the evaluator).
    """

    path: str
    width: int
    height: int
    boxes: npt.NDArray[np.float32]
    classes: npt.NDArray[np.int64]
    crowd: npt.NDArray[np.bool_] = field(default_factory=lambda: np.zeros(0, dtype=np.bool_))

    def __post_init__(self) -> None:
        self.boxes = np.asarray(self.boxes, dtype=np.float32).reshape(-1, 4)
        self.classes = np.asarray(self.classes, dtype=np.int64).reshape(-1)
        if self.crowd.shape[0] != self.boxes.shape[0]:
            self.crowd = np.zeros(self.boxes.shape[0], dtype=np.bool_)
        np.clip(self.boxes[:, 0::2], 0, self.width, out=self.boxes[:, 0::2])
        np.clip(self.boxes[:, 1::2], 0, self.height, out=self.boxes[:, 1::2])

    @property
    def aspect_ratio(self) -> float:
        """``height / width``; used for rectangular batching."""
        return self.height / max(self.width, 1)


def image_size(path: str | Path) -> tuple[int, int]:
    """``(width, height)`` from the file header without decoding pixels."""
    with Image.open(path) as im:
        return int(im.width), int(im.height)


def list_images(directory: str | Path) -> list[Path]:
    """Sorted image files directly under ``directory``."""
    d = Path(directory)
    if not d.is_dir():
        raise FileNotFoundError(f"image directory not found: {d}")
    return sorted(p for p in d.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)
