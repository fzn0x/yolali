from pathlib import Path

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from yolali.data import (
    DetectionDataset,
    RectBatchSampler,
    collate_samples,
    default_train_transforms,
    default_val_transforms,
    read_yolo_txt,
)
from yolali.data.dataset import AugmentConfig


def test_load_raw_resizes_long_side(yolo_dataset: Path):
    records = read_yolo_txt(yolo_dataset / "images" / "train")
    ds = DetectionDataset(records, img_size=64)
    s = ds.load_raw(0)  # 120x160 -> long side 64 -> 48x64
    assert s.hw == (48, 64)
    assert s.orig_hw == (120, 160)
    assert np.allclose(s.boxes, records[0].boxes * 0.4)


def test_getitem_is_deterministic_per_epoch(yolo_dataset: Path):
    records = read_yolo_txt(yolo_dataset / "images" / "train")
    ds = DetectionDataset(records, img_size=64, seed=1)
    ds.transform = default_train_transforms(ds, 64, AugmentConfig(mixup=0.5))
    a, b = ds[1], ds[1]
    assert np.array_equal(a.image, b.image) and np.array_equal(a.boxes, b.boxes)
    ds.set_epoch(1)
    c = ds[1]
    assert not np.array_equal(a.image, c.image)
    assert a.hw == c.hw == (64, 64)


def test_train_pipeline_batches_via_dataloader(yolo_dataset: Path):
    records = read_yolo_txt(yolo_dataset / "images" / "train")
    ds = DetectionDataset(records, img_size=64)
    ds.transform = default_train_transforms(ds, 64)
    loader = DataLoader(ds, batch_size=3, collate_fn=collate_samples)
    batch = next(iter(loader))
    assert batch.images.shape == (3, 3, 64, 64)
    assert batch.images.dtype == torch.float32 and batch.images.max() <= 1.0
    assert batch.targets.boxes.shape[0] == 3
    assert len(batch.paths) == 3 and len(batch) == 3


def test_val_pipeline_letterboxes(yolo_dataset: Path):
    records = read_yolo_txt(yolo_dataset / "images" / "train")
    ds = DetectionDataset(records, img_size=96, transform=default_val_transforms(96))
    s = ds[1]  # 200x100 portrait -> long side 96 -> 96x48 -> padded to 96x96
    assert s.hw == (96, 96)
    assert s.pad == (24.0, 0.0)


def test_rect_batch_sampler_shapes_and_order(yolo_dataset: Path):
    records = read_yolo_txt(yolo_dataset / "images" / "train")
    ars = [r.aspect_ratio for r in records]  # [0.75, 2.0, 1.0]
    sampler = RectBatchSampler(ars, batch_size=1, img_size=96, stride=32, pad=0.0)
    batches = list(sampler)
    assert batches == [[0], [2], [1]]  # sorted by aspect ratio
    # Landscape 0.75 -> h = ceil(0.75*96/32)*32 = 96 (3 blocks), w = 96.
    assert sampler.batch_shapes[0] == (96, 96)
    # Portrait 2.0 -> w = ceil(0.5*96/32)*32 = 64.
    assert sampler.batch_shapes[1] == (96, 64)
    assert len(sampler) == 3


def test_rect_mode_end_to_end(yolo_dataset: Path):
    records = read_yolo_txt(yolo_dataset / "images" / "train")
    sampler = RectBatchSampler([r.aspect_ratio for r in records], 1, 96, pad=0.0)
    ds = DetectionDataset(
        records, img_size=96, transform=default_val_transforms(96), rect_shapes=sampler.batch_shapes
    )
    loader = DataLoader(ds, batch_sampler=sampler, collate_fn=collate_samples)
    shapes = [tuple(b.images.shape[-2:]) for b in loader]
    assert shapes == [(96, 96), (96, 96), (96, 64)]


def test_collate_rejects_mixed_shapes(yolo_dataset: Path):
    records = read_yolo_txt(yolo_dataset / "images" / "train")
    ds = DetectionDataset(records, img_size=64)
    with pytest.raises(ValueError, match="share a shape"):
        collate_samples([ds[0], ds[1]])
