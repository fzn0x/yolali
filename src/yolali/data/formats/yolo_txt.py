"""YOLO txt labels: one ``cls cx cy w h`` line per box, normalised to ``[0, 1]``.

Layout convention: ``.../images/<split>/img.jpg`` pairs with
``.../labels/<split>/img.txt``. A missing label file means "no objects".
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from yolali.data.formats.record import LabelRecord, image_size, list_images


def label_path_for(image_path: Path, labels_dir: Path | None = None) -> Path:
    """``images/x/a.jpg`` -> ``labels/x/a.txt`` (or ``labels_dir/a.txt`` when given)."""
    if labels_dir is not None:
        return labels_dir / (image_path.stem + ".txt")
    parts = list(image_path.parts)
    try:
        i = len(parts) - 1 - parts[::-1].index("images")
    except ValueError as exc:
        raise ValueError(
            f"cannot infer label path for {image_path}: no 'images' directory"
        ) from exc
    parts[i] = "labels"
    return Path(*parts).with_suffix(".txt")


def parse_yolo_lines(text: str, width: int, height: int) -> tuple[np.ndarray, np.ndarray]:
    """Parse label text into ``(boxes xyxy px float32, classes int64)``."""
    rows = [line.split() for line in text.splitlines() if line.strip()]
    if not rows:
        return np.zeros((0, 4), dtype=np.float32), np.zeros(0, dtype=np.int64)
    arr = np.array([[float(v) for v in r[:5]] for r in rows], dtype=np.float64)
    if arr.shape[1] != 5:
        raise ValueError("each YOLO label line needs 5 values: cls cx cy w h")
    classes = arr[:, 0].astype(np.int64)
    cx, cy, w, h = (arr[:, 1] * width, arr[:, 2] * height, arr[:, 3] * width, arr[:, 4] * height)
    boxes = np.stack((cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2), 1).astype(np.float32)
    return boxes, classes


def read_yolo_txt(
    images_dir: str | Path, labels_dir: str | Path | None = None
) -> list[LabelRecord]:
    """Read every image under ``images_dir`` with its YOLO txt label."""
    labels = Path(labels_dir) if labels_dir is not None else None
    records: list[LabelRecord] = []
    for img in list_images(images_dir):
        w, h = image_size(img)
        label = label_path_for(img, labels)
        text = label.read_text(encoding="utf-8") if label.exists() else ""
        boxes, classes = parse_yolo_lines(text, w, h)
        records.append(LabelRecord(str(img.resolve()), w, h, boxes, classes))
    return records
