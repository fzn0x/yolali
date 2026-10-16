"""Evaluate a model on a dataset with yolali's own COCO-protocol evaluator.

COCO val2017 with converted official GELAN-C weights (verification gate 3):

    uv run scripts/convert_yolov9.py weights/gelan-c.pt weights/gelan-c.yolali.pt
    uv run scripts/val.py --config configs/models/gelan-c.yaml \
        --weights weights/gelan-c.yolali.pt --images datasets/coco/val2017 \
        --coco-json datasets/coco/annotations/instances_val2017.json

A YOLO-format dataset (labels next to images, no crowd flags):

    uv run scripts/val.py --config configs/models/gelan-n.yaml --weights runs/voc/best.pt \\
        --images datasets/voc/images/val
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from yolali.data import DetectionDataset, LabelCache, RectBatchSampler, collate_samples
from yolali.data.dataset import default_val_transforms
from yolali.data.formats import LabelRecord, list_images, read_coco_json, read_yolo_txt
from yolali.eval import EvalConfig, Evaluator
from yolali.nn.model import load_model


def gt_lookup_from_records(records: list[LabelRecord]):  # noqa: ANN201 - returns a closure
    """Ground truth in original pixels, with crowd flags and (box) areas."""
    by_path = {r.path: r for r in records}

    def lookup(path: str) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        r = by_path[path]
        boxes = torch.from_numpy(r.boxes)
        areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        return boxes, torch.from_numpy(r.classes), torch.from_numpy(r.crowd), areas

    return lookup


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True, help="yolali state_dict .pt")
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument(
        "--coco-json", type=Path, default=None, help="COCO instances json (else YOLO txt labels)"
    )
    parser.add_argument("--img-size", type=int, default=640)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--conf", type=float, default=0.001)
    parser.add_argument("--iou", type=float, default=0.7)
    parser.add_argument("--max-det", type=int, default=300)
    parser.add_argument("--no-nms", action="store_true", help="NMS-free head: keep top-k directly")
    parser.add_argument(
        "--no-rect", action="store_true", help="square letterbox instead of rect batches"
    )
    parser.add_argument("--half", action="store_true")
    parser.add_argument("--fuse", action="store_true", help="fold BatchNorm before running")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--limit", type=int, default=0, help="evaluate only the first N images")
    parser.add_argument("--out", type=Path, default=None, help="write metrics json here")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.coco_json is not None:
        records, names, _ = read_coco_json(args.coco_json, args.images)
        areas: dict[str, torch.Tensor] = {}
        with args.coco_json.open(encoding="utf-8") as f:
            raw = json.load(f)
        file_by_id = {
            int(im["id"]): str((args.images / im["file_name"]).resolve()) for im in raw["images"]
        }
        per_image: dict[str, list[float]] = {}
        for ann in raw["annotations"]:
            per_image.setdefault(file_by_id[int(ann["image_id"])], []).append(float(ann["area"]))
        for path, vals in per_image.items():
            areas[path] = torch.tensor(vals, dtype=torch.float32)
        nc = len(names)
    else:
        cache = LabelCache(args.images.parent / f"{args.images.name}.cache")
        records = cache.load_or_build(list_images(args.images), lambda: read_yolo_txt(args.images))
        areas = {}
        nc = int(max((int(r.classes.max()) for r in records if len(r.classes)), default=-1)) + 1
    if args.limit:
        records = records[: args.limit]

    model = load_model(args.config, args.weights, device=args.device).eval()
    if args.fuse:
        model.fuse()
    if model.num_classes != nc:
        print(f"warning: model has {model.num_classes} classes, dataset has {nc}", file=sys.stderr)

    rect = not args.no_rect
    sampler = RectBatchSampler([r.aspect_ratio for r in records], args.batch_size, args.img_size)
    ds = DetectionDataset(
        records,
        img_size=args.img_size,
        transform=None if rect else default_val_transforms(args.img_size),
        rect_shapes=sampler.batch_shapes if rect else None,
    )
    loader = (
        DataLoader(ds, batch_sampler=sampler, collate_fn=collate_samples, num_workers=args.workers)
        if rect
        else DataLoader(
            ds, batch_size=args.batch_size, collate_fn=collate_samples, num_workers=args.workers
        )
    )

    base_lookup = gt_lookup_from_records(records)

    def lookup(path: str) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        boxes, classes, crowd, box_areas = base_lookup(path)
        seg_areas = areas.get(path)
        # COCO json annotations are in the same order as read_coco_json emits them.
        return boxes, classes, crowd, seg_areas if seg_areas is not None else box_areas

    cfg = EvalConfig(
        conf=args.conf, iou=args.iou, max_det=args.max_det, nms=not args.no_nms, half=args.half
    )
    evaluator = Evaluator(model, model.num_classes, cfg, device=args.device, gt_lookup=lookup)
    n_batches = len(loader)
    t0 = time.perf_counter()
    metrics = evaluator.run(
        loader,
        progress=lambda n: print(f"\r{n}/{n_batches} batches", end="", file=sys.stderr, flush=True),
    )
    elapsed = time.perf_counter() - t0
    print(file=sys.stderr)
    result = {**metrics.as_dict(), "images": len(records), "seconds": round(elapsed, 1)}
    for k, v in result.items():
        print(f"{k:>12s}: {v:.4f}" if isinstance(v, float) else f"{k:>12s}: {v}")
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
