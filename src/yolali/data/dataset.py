"""``DetectionDataset``: records + transforms -> samples."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from torch.utils.data import Dataset

from yolali.data.formats.record import LabelRecord
from yolali.data.sample import Sample, UInt8Image
from yolali.data.transforms.base import Compose, Transform
from yolali.data.transforms.color import RandomHSV
from yolali.data.transforms.flip import RandomFlip
from yolali.data.transforms.letterbox import Letterbox
from yolali.data.transforms.mixup import MixUp
from yolali.data.transforms.mosaic import Mosaic
from yolali.data.transforms.perspective import RandomPerspective


@dataclass(frozen=True)
class AugmentConfig:
    """Augmentation strengths for :func:`default_train_transforms`."""

    mosaic: float = 1.0
    mixup: float = 0.0
    hsv_h: float = 0.015
    hsv_s: float = 0.7
    hsv_v: float = 0.4
    degrees: float = 0.0
    translate: float = 0.1
    scale: float = 0.5
    shear: float = 0.0
    perspective: float = 0.0
    fliplr: float = 0.5
    flipud: float = 0.0


def load_image(path: str | Path) -> UInt8Image:
    """Read an image as ``(H, W, 3)`` uint8 RGB."""
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(f"could not read image: {path}")
    return np.asarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), dtype=np.uint8)


class DetectionDataset(Dataset[Sample]):
    """Index -> :class:`Sample`.

    Args:
        records: labels, one per image.
        img_size: images are resized so their long side is ``img_size``
            *before* ``transform`` runs (mosaic and letterbox expect this).
        transform: a :class:`Transform`; ``None`` returns the resized sample.
        seed: base seed; each ``__getitem__`` derives its own generator from
            ``seed``, the index and ``epoch`` so runs are reproducible.
        rect_shapes: optional per-index letterbox target ``(H, W)`` from
            :class:`~yolali.data.collate.RectBatchSampler`. When given, the
            dataset letterboxes to that shape *after* ``transform``, so the
            transform must not letterbox itself (use ``transform=None`` for
            plain validation).
    """

    def __init__(
        self,
        records: Sequence[LabelRecord],
        img_size: int = 640,
        transform: Transform | None = None,
        *,
        seed: int = 0,
        rect_shapes: dict[int, tuple[int, int]] | None = None,
    ) -> None:
        self.records = list(records)
        self.img_size = img_size
        self.transform = transform
        self.seed = seed
        self.epoch = 0
        self.rect_shapes = rect_shapes

    def __len__(self) -> int:
        return len(self.records)

    def set_epoch(self, epoch: int) -> None:
        """Change the per-epoch randomness (call from the trainer each epoch)."""
        self.epoch = epoch

    def load_raw(self, index: int) -> Sample:
        """Resize so the long side is ``img_size``; boxes follow; no augmentation."""
        rec = self.records[index]
        image = load_image(rec.path)
        h0, w0 = image.shape[:2]
        r = self.img_size / max(h0, w0)
        boxes = rec.boxes[~rec.crowd]
        classes = rec.classes[~rec.crowd]
        if r != 1.0:
            interp = cv2.INTER_LINEAR if r > 1 else cv2.INTER_AREA
            resized = cv2.resize(image, (round(w0 * r), round(h0 * r)), interpolation=interp)
            image = np.asarray(resized, dtype=np.uint8)
            boxes = boxes * r
        return Sample(
            image, boxes.astype(np.float32), classes, path=rec.path, orig_hw=(h0, w0), scale=r
        )

    def random_raw(self, rng: np.random.Generator) -> Sample:
        """A random :meth:`load_raw` sample (used by mosaic / mixup)."""
        return self.load_raw(int(rng.integers(len(self.records))))

    def rng_for(self, index: int) -> np.random.Generator:
        """Deterministic generator for ``(seed, epoch, index)``."""
        return np.random.default_rng([self.seed, self.epoch, index])

    def __getitem__(self, index: int) -> Sample:
        sample = self.load_raw(index)
        rng = self.rng_for(index)
        if self.transform is not None:
            sample = self.transform(sample, rng)
        if self.rect_shapes is not None:
            sample = Letterbox(self.rect_shapes[index])(sample, rng)
        return sample


def default_train_transforms(
    dataset: DetectionDataset, size: int, aug: AugmentConfig | None = None
) -> Compose:
    """Mosaic -> RandomPerspective -> MixUp -> HSV -> flips, in the YOLOv5 lineage order.

    ``dataset.random_raw`` feeds mosaic; the mixup partner is a full
    mosaic+perspective sample so both halves share geometry statistics.
    """
    cfg = aug or AugmentConfig()
    geometry = Compose(
        [
            Mosaic(size, dataset.random_raw, p=cfg.mosaic),
            RandomPerspective(
                degrees=cfg.degrees,
                translate=cfg.translate,
                scale=cfg.scale,
                shear=cfg.shear,
                perspective=cfg.perspective,
                border=(-size // 2, -size // 2) if cfg.mosaic > 0 else (0, 0),
            ),
            Letterbox(size),
        ]
    )

    def mixup_partner(rng: np.random.Generator) -> Sample:
        return geometry(dataset.random_raw(rng), rng)

    return Compose(
        [
            geometry,
            MixUp(mixup_partner, p=cfg.mixup),
            RandomHSV(cfg.hsv_h, cfg.hsv_s, cfg.hsv_v),
            RandomFlip(cfg.flipud, horizontal=False),
            RandomFlip(cfg.fliplr, horizontal=True),
        ]
    )


def default_val_transforms(size: int, *, rect: bool = False, stride: int = 32) -> Compose:
    """Letterbox only; ``rect=True`` pads to the minimal stride multiple."""
    return Compose([Letterbox(size, stride=stride, auto=rect)])
