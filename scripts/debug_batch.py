"""Render one augmented training batch as an image grid for eyeballing.

Example:
    uv run scripts/debug_batch.py --images datasets/coco128/images/train2017 --n 16
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

from yolali.data import DetectionDataset, LabelCache, default_train_transforms, read_yolo_txt
from yolali.data.dataset import AugmentConfig
from yolali.data.formats import list_images


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images", type=Path, required=True, help="YOLO-format images directory")
    parser.add_argument("--out", type=Path, default=Path("runs/debug/batch.jpg"))
    parser.add_argument("--img-size", type=int, default=640)
    parser.add_argument("--n", type=int, default=16)
    parser.add_argument("--mixup", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    cache = LabelCache(args.images.parent / f"{args.images.name}.cache")
    records = cache.load_or_build(list_images(args.images), lambda: read_yolo_txt(args.images))
    ds = DetectionDataset(records, img_size=args.img_size, seed=args.seed)
    ds.transform = default_train_transforms(ds, args.img_size, AugmentConfig(mixup=args.mixup))

    cols = int(np.ceil(np.sqrt(args.n)))
    rows = int(np.ceil(args.n / cols))
    tile = args.img_size
    grid = np.full((rows * tile, cols * tile, 3), 114, dtype=np.uint8)
    rng = np.random.default_rng(args.seed)
    for k in range(args.n):
        s = ds[int(rng.integers(len(ds)))]
        img = np.ascontiguousarray(s.image)
        for (x1, y1, x2, y2), c in zip(s.boxes.astype(int), s.classes, strict=True):
            colour = tuple(int(v) for v in np.random.default_rng(int(c)).integers(64, 255, 3))
            cv2.rectangle(img, (x1, y1), (x2, y2), colour, 2)
            cv2.putText(
                img, str(int(c)), (x1, max(y1 - 4, 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, colour, 1
            )
        r, c = divmod(k, cols)
        grid[r * tile : (r + 1) * tile, c * tile : (c + 1) * tile] = img
    args.out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(
        str(args.out), cv2.cvtColor(grid, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 85]
    )
    print(f"wrote {args.out} ({rows}x{cols} tiles of {tile}px, {len(ds)} images in dataset)")


if __name__ == "__main__":
    main()
