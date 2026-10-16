"""Our evaluator must reproduce pycocotools to 1e-6 on random data.

Random ground truth with crowd regions, jittered detections with wrong
classes, and noise boxes exercise every branch of the matching rules.
"""

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch

from yolali.eval import MetricAccumulator

pytest.importorskip("pycocotools")
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


def synthetic(
    seed: int, nc: int = 3, n_images: int = 25
) -> tuple[dict[str, Any], list[dict[str, Any]], MetricAccumulator]:
    rng = np.random.default_rng(seed)
    images: list[dict[str, Any]] = []
    anns: list[dict[str, Any]] = []
    dets: list[dict[str, Any]] = []
    acc = MetricAccumulator(num_classes=nc)
    ann_id = 1
    for img_id in range(1, n_images + 1):
        images.append({"id": img_id, "width": 640, "height": 480})
        gt_xyxy: list[list[float]] = []
        gt_cls: list[int] = []
        gt_crowd: list[bool] = []
        gt_area: list[float] = []
        for _ in range(int(rng.integers(0, 6))):
            x, y = float(rng.uniform(0, 500)), float(rng.uniform(0, 380))
            w, h = float(rng.uniform(4, 160)), float(rng.uniform(4, 120))
            c = int(rng.integers(nc))
            crowd = bool(rng.random() < 0.1)
            area = w * h * float(rng.uniform(0.5, 1.0))  # segmentation area < box area
            anns.append(
                {
                    "id": ann_id,
                    "image_id": img_id,
                    "category_id": c + 1,
                    "bbox": [x, y, w, h],
                    "area": area,
                    "iscrowd": int(crowd),
                }
            )
            ann_id += 1
            gt_xyxy.append([x, y, x + w, y + h])
            gt_cls.append(c)
            gt_crowd.append(crowd)
            gt_area.append(area)
        pred_xyxy: list[list[float]] = []
        pred_s: list[float] = []
        pred_c: list[int] = []
        for b, c in zip(gt_xyxy, gt_cls, strict=True):
            for _ in range(int(rng.integers(0, 3))):  # 0-2 jittered copies per GT
                j = rng.normal(0, 8, 4)
                pred_xyxy.append([b[0] + j[0], b[1] + j[1], b[2] + j[2], b[3] + j[3]])
                pred_s.append(float(rng.uniform(0.2, 1.0)))
                pred_c.append(c if rng.random() < 0.85 else int(rng.integers(nc)))
        for _ in range(int(rng.integers(0, 4))):
            x, y = float(rng.uniform(0, 500)), float(rng.uniform(0, 380))
            w, h = float(rng.uniform(4, 160)), float(rng.uniform(4, 120))
            pred_xyxy.append([x, y, x + w, y + h])
            pred_s.append(float(rng.uniform(0.05, 0.6)))
            pred_c.append(int(rng.integers(nc)))
        for b, s, c in zip(pred_xyxy, pred_s, pred_c, strict=True):
            dets.append(
                {
                    "image_id": img_id,
                    "category_id": c + 1,
                    "score": s,
                    "bbox": [b[0], b[1], b[2] - b[0], b[3] - b[1]],
                }
            )
        acc.add_image(
            torch.tensor(pred_xyxy, dtype=torch.float64).reshape(-1, 4),
            torch.tensor(pred_s, dtype=torch.float64),
            torch.tensor(pred_c, dtype=torch.int64),
            torch.tensor(gt_xyxy, dtype=torch.float64).reshape(-1, 4),
            torch.tensor(gt_cls, dtype=torch.int64),
            gt_crowd=torch.tensor(gt_crowd, dtype=torch.bool),
            gt_areas=torch.tensor(gt_area, dtype=torch.float64),
        )
    dataset = {
        "images": images,
        "annotations": anns,
        "categories": [{"id": i + 1, "name": str(i)} for i in range(nc)],
    }
    return dataset, dets, acc


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_matches_pycocotools_on_random_data(seed: int, tmp_path: Path):
    dataset, dets, acc = synthetic(seed)
    gt_path, dt_path = tmp_path / "gt.json", tmp_path / "dt.json"
    gt_path.write_text(json.dumps(dataset))
    dt_path.write_text(json.dumps(dets))
    coco = COCO(str(gt_path))
    ev = COCOeval(coco, coco.loadRes(str(dt_path)), "bbox")
    ev.evaluate()
    ev.accumulate()
    ev.summarize()
    ours = acc.compute()
    ref: list[float] = [float(v) for v in ev.stats]
    assert ours.map50_95 == pytest.approx(ref[0], abs=1e-6)
    assert ours.map50 == pytest.approx(ref[1], abs=1e-6)
    assert ours.map75 == pytest.approx(ref[2], abs=1e-6)
    assert ours.map_small == pytest.approx(ref[3], abs=1e-6)
    assert ours.map_medium == pytest.approx(ref[4], abs=1e-6)
    assert ours.map_large == pytest.approx(ref[5], abs=1e-6)
