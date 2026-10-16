import math

import pytest
import torch

from yolali.ops import decode_boxes, dfl_expectation, direct_distances, make_anchors


def test_dfl_expectation_of_one_hot_logits_is_the_bin_index():
    reg_max = 4
    # Side distances 0, 1, 2, 3: put a huge logit on that bin.
    raw = torch.full((1, 1, 4 * reg_max), -1e4)
    for side, k in enumerate([0, 1, 2, 3]):
        raw[0, 0, side * reg_max + k] = 0.0
    assert torch.allclose(dfl_expectation(raw, reg_max), torch.tensor([[[0.0, 1.0, 2.0, 3.0]]]))


def test_dfl_expectation_of_two_equal_bins_is_their_midpoint():
    reg_max = 16
    raw = torch.zeros(1, 1, 4 * reg_max)
    # Equal mass on bins 2 and 3 for every side -> 2.5. Other bins get ~e^-20.
    for side in range(4):
        raw[0, 0, side * reg_max + 2] = 20.0
        raw[0, 0, side * reg_max + 3] = 20.0
    assert torch.allclose(dfl_expectation(raw, reg_max), torch.full((1, 1, 4), 2.5), atol=1e-5)


def test_direct_distances_are_softplus():
    raw = torch.tensor([[[0.0, -1.0, 1.0, 10.0]]])
    out = direct_distances(raw)
    assert out[0, 0, 0].item() == pytest.approx(math.log(2))
    assert (out >= 0).all()
    assert out[0, 0, 3].item() == pytest.approx(10.0, abs=1e-3)


def test_decode_direct_to_pixels():
    anchors, strides = make_anchors([(1, 1)], [8])  # one anchor at (4, 4)
    # softplus^-1(0.5) so each side is 0.5 stride = 4 px -> box [0, 0, 8, 8]
    raw = torch.full((1, 1, 4), math.log(math.exp(0.5) - 1))
    boxes = decode_boxes(raw, anchors, strides, reg_max=0)
    assert torch.allclose(boxes, torch.tensor([[[0.0, 0.0, 8.0, 8.0]]]), atol=1e-5)


def test_decode_dfl_to_pixels():
    anchors, strides = make_anchors([(1, 1)], [16])  # anchor at (8, 8)
    reg_max = 16
    raw = torch.full((1, 1, 4 * reg_max), -1e4)
    # l=1, t=2, r=3, b=4 stride units -> x1 = 8-16, y1 = 8-32, x2 = 8+48, y2 = 8+64
    for side, k in enumerate([1, 2, 3, 4]):
        raw[0, 0, side * reg_max + k] = 0.0
    boxes = decode_boxes(raw, anchors, strides, reg_max=reg_max)
    assert torch.allclose(boxes, torch.tensor([[[-8.0, -24.0, 56.0, 72.0]]]))
