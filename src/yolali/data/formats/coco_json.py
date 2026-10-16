"""COCO ``instances_*.json`` labels.

Category ids are mapped to contiguous indices in ascending id order (COCO's
91-id space becomes 0..79). ``iscrowd`` regions are kept and flagged so the
evaluator can ignore detections that match them.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from yolali.data.formats.record import LabelRecord


def read_coco_json(
    annotations: str | Path, images_dir: str | Path, *, include_empty: bool = True
) -> tuple[list[LabelRecord], list[str], dict[int, int]]:
    """Read a COCO instances file.

    Args:
        annotations: path to ``instances_*.json``.
        images_dir: directory containing the ``file_name`` entries.
        include_empty: keep images that have no annotations.

    Returns:
        ``records``, ``names`` (contiguous index -> category name), and
        ``cat_id_to_index`` (COCO category id -> contiguous index).
    """
    with Path(annotations).open(encoding="utf-8") as f:
        data: dict[str, Any] = json.load(f)
    cats: list[dict[str, Any]] = sorted(data["categories"], key=lambda c: int(c["id"]))
    cat_id_to_index = {int(c["id"]): i for i, c in enumerate(cats)}
    names = [str(c["name"]) for c in cats]

    per_image: dict[int, list[dict[str, Any]]] = {}
    for ann in data["annotations"]:
        per_image.setdefault(int(ann["image_id"]), []).append(ann)

    root = Path(images_dir)
    records: list[LabelRecord] = []
    for img in sorted(data["images"], key=lambda im: int(im["id"])):
        anns = per_image.get(int(img["id"]), [])
        if not anns and not include_empty:
            continue
        w, h = int(img["width"]), int(img["height"])
        boxes = np.zeros((len(anns), 4), dtype=np.float32)
        classes = np.zeros(len(anns), dtype=np.int64)
        crowd = np.zeros(len(anns), dtype=np.bool_)
        for i, ann in enumerate(anns):
            x, y, bw, bh = (float(v) for v in ann["bbox"])
            boxes[i] = (x, y, x + bw, y + bh)
            classes[i] = cat_id_to_index[int(ann["category_id"])]
            crowd[i] = bool(ann.get("iscrowd", 0))
        records.append(
            LabelRecord(str((root / str(img["file_name"])).resolve()), w, h, boxes, classes, crowd)
        )
    return records, names, cat_id_to_index
