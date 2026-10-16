"""Run a detector on images and print / save the results.

    uv run scripts/detect.py --config configs/models/gelan-c.yaml \\
        --weights weights/gelan-c.yolali.pt --names configs/data/coco128.yaml \\
        --source datasets/coco128/images/train2017/000000000009.jpg

``--source`` may be an image, a directory of images, or several paths.
Annotated copies go to ``--out`` (default ``runs/detect``).
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2
import yaml

from yolali.data.formats import IMAGE_SUFFIXES
from yolali.infer import Detector, draw_detections
from yolali.infer.preprocess import to_rgb_array
from yolali.nn.model import load_model


def expand_sources(sources: list[Path]) -> list[Path]:
    files: list[Path] = []
    for s in sources:
        if s.is_dir():
            files.extend(p for p in sorted(s.iterdir()) if p.suffix.lower() in IMAGE_SUFFIXES)
        else:
            files.append(s)
    if not files:
        raise SystemExit("no images found")
    return files


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--source", type=Path, nargs="+", required=True)
    parser.add_argument("--names", type=Path, default=None, help="data yaml with a `names` list")
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.7)
    parser.add_argument("--max-det", type=int, default=300)
    parser.add_argument("--img-size", type=int, default=640)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--half", action="store_true")
    parser.add_argument("--out", type=Path, default=Path("runs/detect"))
    parser.add_argument("--no-save", action="store_true")
    args = parser.parse_args()

    names: list[str] | None = None
    if args.names is not None:
        with args.names.open(encoding="utf-8") as f:
            names = [str(n) for n in yaml.safe_load(f)["names"]]
    model = load_model(args.config, args.weights)
    detector = Detector(model, device=args.device, half=args.half, names=names)
    files = expand_sources(args.source)

    t0 = time.perf_counter()
    results = detector.predict(
        files, conf=args.conf, iou=args.iou, max_det=args.max_det, size=args.img_size
    )
    elapsed_ms = (time.perf_counter() - t0) * 1000
    args.out.mkdir(parents=True, exist_ok=True)
    for path, dets in zip(files, results, strict=True):
        summary = ", ".join(f"{dets.label(i)} {float(dets.conf[i]):.2f}" for i in range(len(dets)))
        noun = "object" if len(dets) == 1 else "objects"
        print(f"{path.name}: {len(dets)} {noun}" + (f" ({summary})" if summary else ""))
        if not args.no_save:
            drawn = draw_detections(to_rgb_array(path), dets)
            cv2.imwrite(str(args.out / path.name), cv2.cvtColor(drawn, cv2.COLOR_RGB2BGR))
    device = detector.device.type
    per_image = elapsed_ms / len(files)
    print(f"{len(files)} images in {elapsed_ms:.0f} ms on {device} ({per_image:.1f} ms/image)")
    if not args.no_save:
        print(f"annotated images saved to {args.out}")


if __name__ == "__main__":
    main()
