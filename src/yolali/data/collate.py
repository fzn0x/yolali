"""Samples -> tensors, and rectangular batching."""

from __future__ import annotations

import math
from collections.abc import Iterator, Sequence
from dataclasses import dataclass

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Sampler

from yolali.data.sample import Sample
from yolali.types import Targets


@dataclass(frozen=True)
class Batch:
    """What the model and loss consume.

    Attributes:
        images: ``(B, 3, H, W)`` float32 in ``[0, 1]``, RGB.
        targets: padded ground truth in letterboxed pixels.
        paths: image paths.
        orig_hw: original ``(H, W)`` per image.
        scales: letterbox scale per image.
        pads: letterbox ``(left, top)`` per image.
    """

    images: Tensor
    targets: Targets
    paths: tuple[str, ...]
    orig_hw: tuple[tuple[int, int], ...]
    scales: tuple[float, ...]
    pads: tuple[tuple[float, float], ...]

    def to(self, device: torch.device | str, non_blocking: bool = False) -> Batch:
        """Move tensors to ``device``."""
        return Batch(
            self.images.to(device, non_blocking=non_blocking),
            self.targets.to(device),
            self.paths,
            self.orig_hw,
            self.scales,
            self.pads,
        )

    def __len__(self) -> int:
        return int(self.images.shape[0])


def collate_samples(samples: Sequence[Sample]) -> Batch:
    """Stack equally sized samples into a :class:`Batch`."""
    shapes = {s.hw for s in samples}
    if len(shapes) != 1:
        raise ValueError(f"all samples in a batch must share a shape, got {sorted(shapes)}")
    images = torch.from_numpy(np.stack([s.image for s in samples])).permute(0, 3, 1, 2)
    images = images.contiguous().float().div_(255.0)
    targets = Targets.from_list(
        [torch.from_numpy(s.boxes) for s in samples], [torch.from_numpy(s.classes) for s in samples]
    )
    return Batch(
        images=images,
        targets=targets,
        paths=tuple(s.path for s in samples),
        orig_hw=tuple(s.orig_hw for s in samples),
        scales=tuple(s.scale for s in samples),
        pads=tuple(s.pad for s in samples),
    )


class RectBatchSampler(Sampler[list[int]]):
    """Group images by aspect ratio so each batch letterboxes to a tight rectangle.

    Args:
        aspect_ratios: ``height / width`` per dataset index.
        batch_size: images per batch (the last batch may be smaller).
        img_size: long side in pixels.
        stride: batch shapes are multiples of this.
        pad: extra padding fraction (``0.5`` matches the YOLOv5 lineage).
        shuffle: shuffle the order of *batches* each epoch; images inside a
            batch stay aspect-sorted.
        seed: shuffle seed.

    After construction, ``batch_shapes[index]`` gives the ``(H, W)`` that the
    dataset should letterbox index ``index`` to; pass it as ``rect_shapes``.
    """

    def __init__(
        self,
        aspect_ratios: Sequence[float],
        batch_size: int,
        img_size: int,
        *,
        stride: int = 32,
        pad: float = 0.5,
        shuffle: bool = False,
        seed: int = 0,
    ) -> None:
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.seed = seed
        self.epoch = 0
        order = np.argsort(np.asarray(aspect_ratios, dtype=np.float64), kind="stable")
        self.batches: list[list[int]] = [
            [int(i) for i in order[k : k + batch_size]] for k in range(0, len(order), batch_size)
        ]
        self.batch_shapes: dict[int, tuple[int, int]] = {}
        for batch in self.batches:
            ars = [float(aspect_ratios[i]) for i in batch]
            lo, hi = min(ars), max(ars)
            shape = (1.0, 1.0)  # (h, w) relative to img_size
            if hi < 1:
                shape = (hi, 1.0)
            elif lo > 1:
                shape = (1.0, 1.0 / lo)
            h = int(math.ceil(shape[0] * img_size / stride + pad) * stride)
            w = int(math.ceil(shape[1] * img_size / stride + pad) * stride)
            h, w = (
                min(h, int(math.ceil(img_size / stride + pad) * stride)),
                min(w, int(math.ceil(img_size / stride + pad) * stride)),
            )
            for i in batch:
                self.batch_shapes[i] = (h, w)

    def set_epoch(self, epoch: int) -> None:
        """Reseed the batch shuffle."""
        self.epoch = epoch

    def __iter__(self) -> Iterator[list[int]]:
        order = list(range(len(self.batches)))
        if self.shuffle:
            np.random.default_rng([self.seed, self.epoch]).shuffle(order)
        for k in order:
            yield self.batches[k]

    def __len__(self) -> int:
        return len(self.batches)
