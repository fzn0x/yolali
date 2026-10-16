"""Invariants every transform must keep, plus exact letterbox arithmetic."""

import numpy as np
import pytest
import torch

from yolali.data.sample import Sample
from yolali.data.transforms import (
    Compose,
    Letterbox,
    MixUp,
    Mosaic,
    RandomFlip,
    RandomHSV,
    RandomPerspective,
    undo_letterbox,
)
from yolali.data.transforms.letterbox import letterbox_params


def boxes_inside(sample: Sample) -> bool:
    h, w = sample.hw
    b = sample.boxes
    return bool(
        (b[:, 0] >= 0).all()
        and (b[:, 1] >= 0).all()
        and (b[:, 2] <= w).all()
        and (b[:, 3] <= h).all()
    )


def test_letterbox_params_landscape_into_square():
    # 120x160 -> 320x320: scale 2, new 240x320, pad left 0, top 40.
    scale, (nh, nw), (pl, pt) = letterbox_params((120, 160), (320, 320))
    assert scale == 2.0 and (nh, nw) == (240, 320) and (pl, pt) == (0.0, 40.0)


def test_letterbox_moves_boxes_exactly(sample: Sample):
    out = Letterbox(320)(sample, np.random.default_rng(0))
    assert out.hw == (320, 320)
    assert out.scale == 2.0 and out.pad == (0.0, 40.0)
    expected = sample.boxes * 2 + np.array([0, 40, 0, 40], dtype=np.float32)
    assert np.allclose(out.boxes, expected)
    # Padding rows are gray.
    assert (out.image[:40] == 114).all() and (out.image[-40:] == 114).all()


def test_letterbox_auto_pads_to_stride_multiple(sample: Sample):
    out = Letterbox(320, auto=True, stride=32)(sample, np.random.default_rng(0))
    assert out.hw == (256, 320)  # 240 rounded up to 256
    assert out.pad == (0.0, 8.0)


def test_letterbox_no_scale_up(sample: Sample):
    out = Letterbox(320, scale_up=False)(sample, np.random.default_rng(0))
    assert out.scale == 1.0
    assert out.hw == (320, 320)


def test_undo_letterbox_inverts_within_half_pixel(sample: Sample):
    out = Letterbox(416)(sample, np.random.default_rng(0))
    back = undo_letterbox(torch.from_numpy(out.boxes), out.scale, out.pad, sample.orig_hw)
    assert torch.allclose(back, torch.from_numpy(sample.boxes), atol=0.5)


def test_hflip_mirrors_boxes(sample: Sample):
    out = RandomFlip(p=1.0)(sample, np.random.default_rng(0))
    w = sample.hw[1]
    assert np.allclose(out.boxes[:, 0], w - sample.boxes[:, 2])
    assert np.allclose(out.boxes[:, 2], w - sample.boxes[:, 0])
    assert np.array_equal(out.image, sample.image[:, ::-1])


def test_vflip_mirrors_boxes(sample: Sample):
    out = RandomFlip(p=1.0, horizontal=False)(sample, np.random.default_rng(0))
    h = sample.hw[0]
    assert np.allclose(out.boxes[:, 1], h - sample.boxes[:, 3])


def test_flip_probability_zero_is_identity(sample: Sample):
    out = RandomFlip(p=0.0)(sample, np.random.default_rng(0))
    assert out is sample


def test_hsv_keeps_shape_and_boxes(sample: Sample):
    out = RandomHSV()(sample, np.random.default_rng(1))
    assert out.image.shape == sample.image.shape and out.image.dtype == np.uint8
    assert np.array_equal(out.boxes, sample.boxes)
    assert not np.array_equal(out.image, sample.image)


def test_hsv_zero_gains_is_identity(sample: Sample):
    assert RandomHSV(0, 0, 0)(sample, np.random.default_rng(0)) is sample


@pytest.mark.parametrize("seed", range(5))
def test_perspective_keeps_boxes_inside_and_classes_aligned(sample: Sample, seed: int):
    t = RandomPerspective(degrees=10, translate=0.1, scale=0.5, shear=5, perspective=0.0005)
    out = t(sample, np.random.default_rng(seed))
    assert out.hw == sample.hw
    assert boxes_inside(out)
    assert len(out.boxes) == len(out.classes) <= len(sample.boxes)
    assert set(out.classes.tolist()) <= set(sample.classes.tolist())


def test_perspective_border_crops(sample: Sample):
    out = RandomPerspective(border=(-10, -20))(sample, np.random.default_rng(0))
    assert out.hw == (100, 120)


def test_perspective_identity_when_all_zero(sample: Sample):
    out = RandomPerspective(degrees=0, translate=0, scale=0, shear=0, perspective=0)(
        sample, np.random.default_rng(0)
    )
    assert out is sample


def make_mosaic_sample(seed: int) -> Sample:
    rng = np.random.default_rng(seed)
    h, w = int(rng.integers(60, 120)), int(rng.integers(60, 120))
    boxes = np.array([[5, 5, w - 5, h - 5]], dtype=np.float32)
    return Sample(np.full((h, w, 3), seed * 20, dtype=np.uint8), boxes, np.array([seed]))


def test_mosaic_canvas_and_labels():
    size = 128
    partners = iter(make_mosaic_sample(s) for s in (1, 2, 3))
    mosaic = Mosaic(size, sampler=lambda rng: next(partners), p=1.0)
    out = mosaic(make_mosaic_sample(0), np.random.default_rng(0))
    assert out.hw == (2 * size, 2 * size)
    assert boxes_inside(out)
    assert len(out.boxes) == len(out.classes)
    assert set(out.classes.tolist()) <= {0, 1, 2, 3}
    assert len(out.boxes) >= 1


def test_mosaic_then_perspective_returns_to_size():
    size = 128
    partners = iter(make_mosaic_sample(s) for s in (1, 2, 3))
    pipe = Compose(
        [
            Mosaic(size, sampler=lambda rng: next(partners)),
            RandomPerspective(scale=0.5, border=(-size // 2, -size // 2)),
        ]
    )
    out = pipe(make_mosaic_sample(0), np.random.default_rng(3))
    assert out.hw == (size, size)
    assert boxes_inside(out)


def test_mixup_blends_and_concatenates(sample: Sample):
    other = sample.with_(image=np.zeros_like(sample.image), classes=np.array([7, 8]))
    out = MixUp(lambda rng: other, p=1.0, alpha=1e6)(sample, np.random.default_rng(0))  # r ~ 0.5
    assert out.hw == sample.hw
    assert np.allclose(out.image.astype(int), sample.image.astype(int) // 2, atol=1)
    assert out.classes.tolist() == [1, 3, 7, 8]
    assert out.boxes.shape == (4, 4)


def test_mixup_requires_matching_shape(sample: Sample):
    other = Sample(
        np.zeros((10, 10, 3), dtype=np.uint8),
        np.zeros((0, 4), dtype=np.float32),
        np.zeros(0, dtype=np.int64),
    )
    with pytest.raises(ValueError, match="letterbox both"):
        MixUp(lambda rng: other, p=1.0)(sample, np.random.default_rng(0))


def test_compose_repr(sample: Sample):
    c = Compose([Letterbox(64), RandomFlip()])
    assert repr(c) == "Compose([Letterbox, RandomFlip])"
    assert c(sample, np.random.default_rng(0)).hw == (64, 64)


def test_double_letterbox_composes_scale_and_pad(sample: Sample):
    rng = np.random.default_rng(0)
    once = Letterbox(320)(sample, rng)  # 120x160 -> 240x320 + pad top 40
    twice = Letterbox((160, 160))(once, rng)  # 320x320 -> 160x160, scale .5
    assert twice.scale == pytest.approx(1.0)
    assert twice.pad == pytest.approx((0.0, 20.0))
    back = undo_letterbox(torch.from_numpy(twice.boxes), twice.scale, twice.pad, sample.orig_hw)
    assert torch.allclose(back, torch.from_numpy(sample.boxes), atol=0.5)
