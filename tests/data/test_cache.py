import os
import time
from pathlib import Path

import numpy as np

from yolali.data.cache import LabelCache, fingerprint
from yolali.data.formats import LabelRecord, list_images, read_yolo_txt


def test_cache_roundtrip_and_hit(yolo_dataset: Path):
    images = list_images(yolo_dataset / "images" / "train")
    cache = LabelCache(yolo_dataset / "labels.cache")
    calls = 0

    def build() -> list[LabelRecord]:
        nonlocal calls
        calls += 1
        return read_yolo_txt(yolo_dataset / "images" / "train")

    first = cache.load_or_build(images, build)
    second = cache.load_or_build(images, build)
    assert calls == 1
    assert len(second) == 3
    for a, b in zip(first, second, strict=True):
        assert (
            a.path == b.path
            and np.array_equal(a.boxes, b.boxes)
            and np.array_equal(a.classes, b.classes)
        )
        assert b.boxes.dtype == np.float32 and b.classes.dtype == np.int64


def test_cache_invalidates_on_mtime_change(yolo_dataset: Path):
    images = list_images(yolo_dataset / "images" / "train")
    cache = LabelCache(yolo_dataset / "labels.cache")
    calls = 0

    def build() -> list[LabelRecord]:
        nonlocal calls
        calls += 1
        return read_yolo_txt(yolo_dataset / "images" / "train")

    cache.load_or_build(images, build)
    future = time.time() + 10
    os.utime(images[0], (future, future))
    cache.load_or_build(images, build)
    assert calls == 2


def test_fingerprint_depends_on_extra(yolo_dataset: Path):
    images = list_images(yolo_dataset / "images" / "train")
    assert fingerprint(images, "a") != fingerprint(images, "b")
    assert fingerprint(images) == fingerprint(list(reversed(images)))


def test_corrupt_cache_is_rebuilt(yolo_dataset: Path):
    path = yolo_dataset / "labels.cache"
    path.write_bytes(b"not a pickle")
    records = LabelCache(path).load_or_build(
        list_images(yolo_dataset / "images" / "train"),
        lambda: read_yolo_txt(yolo_dataset / "images" / "train"),
    )
    assert len(records) == 3
