"""Draw detections on an image (OpenCV)."""

from __future__ import annotations

import cv2
import numpy as np

from yolali.infer.detections import Detections


def class_colour(index: int) -> tuple[int, int, int]:
    """A stable, reasonably distinct RGB colour per class index."""
    rng = np.random.default_rng(index + 7)
    r, g, b = (int(v) for v in rng.integers(60, 230, 3))
    return r, g, b


def draw_detections(image: np.ndarray, dets: Detections, *, thickness: int = 2) -> np.ndarray:
    """Return a copy of ``image`` (``(H, W, 3)`` uint8 RGB) with boxes and labels drawn."""
    out = np.ascontiguousarray(image.copy())
    for i, (box, conf, c) in enumerate(dets):
        x1, y1, x2, y2 = (round(float(v)) for v in box)
        colour = class_colour(c)
        cv2.rectangle(out, (x1, y1), (x2, y2), colour, thickness)
        text = f"{dets.label(i)} {conf:.2f}"
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        y_text = max(y1 - 4, th + 4)
        cv2.rectangle(out, (x1, y_text - th - 4), (x1 + tw + 4, y_text + 2), colour, -1)
        cv2.putText(
            out, text, (x1 + 2, y_text - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1
        )
    return out
