"""Four-image mosaic."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from yolali.data.sample import Sample
from yolali.data.transforms.letterbox import GRAY

Sampler = Callable[[np.random.Generator], Sample]


class Mosaic:
    """Tile the sample with three others on a ``2s x 2s`` canvas.

    The four images meet at a random centre in ``[s/2, 3s/2]²``; each is
    placed at its native resolution (after the dataset's own resize to
    ``s`` on the long side) and cropped at the canvas edge. Follow with
    :class:`RandomPerspective` using ``border=(-s/2, -s/2)`` to get back
    to ``s x s`` with scale augmentation, as in the YOLOv5 lineage.

    Args:
        size: ``s``.
        sampler: returns another (untransformed) sample; the dataset provides it.
        p: probability of applying the mosaic.
    """

    def __init__(self, size: int, sampler: Sampler, p: float = 1.0) -> None:
        self.size = size
        self.sampler = sampler
        self.p = p

    def __call__(self, sample: Sample, rng: np.random.Generator) -> Sample:
        """Build the canvas; boxes are offset to canvas coordinates and clipped."""
        if rng.random() >= self.p:
            return sample
        s = self.size
        canvas = np.full((2 * s, 2 * s, 3), GRAY, dtype=np.uint8)
        xc, yc = (int(rng.uniform(s // 2, 3 * s // 2)) for _ in range(2))
        boxes: list[np.ndarray] = []
        classes: list[np.ndarray] = []

        for i in range(4):
            cur = sample if i == 0 else self.sampler(rng)
            h, w = cur.hw
            # Canvas region (x1a, y1a, x2a, y2a) and matching image crop (x1b, y1b, x2b, y2b).
            if i == 0:  # top-left
                x1a, y1a, x2a, y2a = max(xc - w, 0), max(yc - h, 0), xc, yc
                x1b, y1b, x2b, y2b = w - (x2a - x1a), h - (y2a - y1a), w, h
            elif i == 1:  # top-right
                x1a, y1a, x2a, y2a = xc, max(yc - h, 0), min(xc + w, 2 * s), yc
                x1b, y1b, x2b, y2b = 0, h - (y2a - y1a), min(w, x2a - x1a), h
            elif i == 2:  # bottom-left
                x1a, y1a, x2a, y2a = max(xc - w, 0), yc, xc, min(2 * s, yc + h)
                x1b, y1b, x2b, y2b = w - (x2a - x1a), 0, w, min(y2a - y1a, h)
            else:  # bottom-right
                x1a, y1a, x2a, y2a = xc, yc, min(xc + w, 2 * s), min(2 * s, yc + h)
                x1b, y1b, x2b, y2b = 0, 0, min(w, x2a - x1a), min(y2a - y1a, h)
            canvas[y1a:y2a, x1a:x2a] = cur.image[y1b:y2b, x1b:x2b]
            dx, dy = x1a - x1b, y1a - y1b
            if len(cur.boxes):
                b = cur.boxes.copy()
                b[:, [0, 2]] += dx
                b[:, [1, 3]] += dy
                boxes.append(b)
                classes.append(cur.classes)

        if boxes:
            all_boxes = np.concatenate(boxes).clip(0, 2 * s).astype(np.float32)
            all_classes = np.concatenate(classes)
            wh = all_boxes[:, 2:] - all_boxes[:, :2]
            keep = (wh > 1).all(1)
            all_boxes, all_classes = all_boxes[keep], all_classes[keep]
        else:
            all_boxes = np.zeros((0, 4), dtype=np.float32)
            all_classes = np.zeros(0, dtype=np.int64)
        return sample.with_(image=canvas, boxes=all_boxes, classes=all_classes)
