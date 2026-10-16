"""Synthetic images and labels for data-pipeline tests (no downloads)."""

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from yolali.data.sample import Sample


def draw_rects(h: int, w: int, boxes: np.ndarray, seed: int = 0) -> np.ndarray:
    """A gray image with a coloured rectangle per box, so boxes are visually meaningful."""
    rng = np.random.default_rng(seed)
    img = np.full((h, w, 3), 90, dtype=np.uint8)
    for x1, y1, x2, y2 in boxes.astype(int):
        img[y1:y2, x1:x2] = rng.integers(0, 255, 3, dtype=np.uint8)
    return img


@pytest.fixture
def sample() -> Sample:
    boxes = np.array([[10, 20, 60, 80], [100, 30, 150, 110]], dtype=np.float32)
    return Sample(draw_rects(120, 160, boxes), boxes, np.array([1, 3]), path="synthetic")


@pytest.fixture
def yolo_dataset(tmp_path: Path) -> Path:
    """``root/images/train/*.jpg`` + ``root/labels/train/*.txt`` with three images."""
    root = tmp_path / "ds"
    (root / "images" / "train").mkdir(parents=True)
    (root / "labels" / "train").mkdir(parents=True)
    specs: list[tuple[int, int, list[tuple[int, float, float, float, float]]]] = [
        (120, 160, [(0, 0.25, 0.5, 0.3, 0.4)]),
        (200, 100, [(2, 0.5, 0.5, 0.2, 0.2), (1, 0.7, 0.3, 0.1, 0.1)]),
        (90, 90, []),
    ]
    for i, (h, w, rows) in enumerate(specs):
        img = np.full((h, w, 3), 120, dtype=np.uint8)
        Image.fromarray(img).save(root / "images" / "train" / f"img{i}.jpg")
        if rows:
            text = "\n".join(" ".join(str(v) for v in r) for r in rows)
            (root / "labels" / "train" / f"img{i}.txt").write_text(text + "\n")
    return root
