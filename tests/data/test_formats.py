import json
from pathlib import Path

import numpy as np
import pytest

from yolali.data.formats import LabelRecord, read_coco_json, read_yolo_txt
from yolali.data.formats.yolo_txt import label_path_for, parse_yolo_lines


def test_parse_yolo_lines_converts_to_pixels():
    boxes, classes = parse_yolo_lines("3 0.5 0.5 0.5 0.25\n", width=200, height=100)
    assert classes.tolist() == [3]
    # cx=100, cy=50, w=100, h=25 -> [50, 37.5, 150, 62.5]
    assert np.allclose(boxes, [[50, 37.5, 150, 62.5]])


def test_parse_yolo_lines_empty():
    boxes, classes = parse_yolo_lines("", 10, 10)
    assert boxes.shape == (0, 4) and classes.shape == (0,)


def test_label_path_swaps_images_for_labels():
    p = label_path_for(Path("/data/images/train/a.jpg"))
    assert p.as_posix().endswith("data/labels/train/a.txt")


def test_label_path_explicit_dir():
    assert label_path_for(Path("x/a.png"), Path("lbl")) == Path("lbl/a.txt")


def test_read_yolo_txt(yolo_dataset: Path):
    records = read_yolo_txt(yolo_dataset / "images" / "train")
    assert len(records) == 3
    assert [r.width for r in records] == [160, 100, 90]
    assert records[0].classes.tolist() == [0]
    # cx .25*160=40, cy .5*120=60, w .3*160=48, h .4*120=48 -> [16, 36, 64, 84]
    assert np.allclose(records[0].boxes, [[16, 36, 64, 84]])
    assert records[1].classes.tolist() == [2, 1]
    assert records[2].boxes.shape == (0, 4)


def test_read_coco_json_matches_yolo(yolo_dataset: Path, tmp_path: Path):
    yolo = read_yolo_txt(yolo_dataset / "images" / "train")
    # Write the same labels as COCO json with non-contiguous category ids.
    cat_ids = {0: 1, 1: 5, 2: 9}
    images, annotations = [], []
    for i, r in enumerate(yolo):
        images.append(
            {"id": i + 1, "file_name": Path(r.path).name, "width": r.width, "height": r.height}
        )
        for b, c in zip(r.boxes, r.classes, strict=True):
            x1, y1, x2, y2 = (float(v) for v in b)
            annotations.append(
                {
                    "id": len(annotations),
                    "image_id": i + 1,
                    "category_id": cat_ids[int(c)],
                    "bbox": [x1, y1, x2 - x1, y2 - y1],
                    "iscrowd": 0,
                    "area": (x2 - x1) * (y2 - y1),
                }
            )
    categories = [{"id": v, "name": f"c{k}"} for k, v in cat_ids.items()]
    ann_path = tmp_path / "ann.json"
    ann_path.write_text(
        json.dumps({"images": images, "annotations": annotations, "categories": categories})
    )

    coco, names, mapping = read_coco_json(ann_path, yolo_dataset / "images" / "train")
    assert names == ["c0", "c1", "c2"]
    assert mapping == {1: 0, 5: 1, 9: 2}
    assert len(coco) == 3
    for a, b in zip(yolo, coco, strict=True):
        assert a.path == b.path
        assert np.allclose(a.boxes, b.boxes, atol=1e-4)
        assert a.classes.tolist() == b.classes.tolist()
    without_empty, _, _ = read_coco_json(
        ann_path, yolo_dataset / "images" / "train", include_empty=False
    )
    assert len(without_empty) == 2


def test_record_clips_boxes_to_image():
    r = LabelRecord("x", 10, 10, np.array([[-5, -5, 20, 8]], dtype=np.float32), np.array([0]))
    assert r.boxes.tolist() == [[0, 0, 10, 8]]
    assert r.crowd.tolist() == [False]
    assert r.aspect_ratio == pytest.approx(1.0)
