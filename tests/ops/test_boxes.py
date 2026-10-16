"""Hand-computed IoU family and format conversions.

Every expected number below is derived in the comment next to it so a reader
can check the arithmetic without running anything.
"""

import math

import pytest
import torch

from yolali.ops import (
    box_iou,
    cxcywh_to_xyxy,
    elementwise_iou,
    ltrb_to_xyxy,
    xyxy_to_cxcywh,
    xyxy_to_ltrb,
)

A = torch.tensor([[0.0, 0.0, 10.0, 10.0]])  # 10x10 square at origin, area 100
B = torch.tensor([[5.0, 5.0, 15.0, 15.0]])  # same square shifted by (5, 5)
C = torch.tensor([[20.0, 0.0, 30.0, 10.0]])  # disjoint from A, same row
D = torch.tensor([[0.0, 0.0, 20.0, 5.0]])  # 20x5, aspect differs from A


def test_iou_overlapping_squares():
    # inter = 5*5 = 25, union = 100 + 100 - 25 = 175 -> 1/7
    assert elementwise_iou(A, B, "iou").item() == pytest.approx(1 / 7, abs=1e-6)


def test_iou_identical_and_disjoint():
    assert elementwise_iou(A, A, "iou").item() == pytest.approx(1.0, abs=1e-6)
    assert elementwise_iou(A, C, "iou").item() == pytest.approx(0.0, abs=1e-6)


def test_giou_overlapping_squares():
    # enclosing box [0,0,15,15] area 225; giou = 1/7 - (225-175)/225
    expected = 1 / 7 - 50 / 225
    assert elementwise_iou(A, B, "giou").item() == pytest.approx(expected, abs=1e-6)


def test_giou_disjoint_is_negative():
    # enclosing [0,0,30,10] = 300, union = 200 -> 0 - 100/300
    assert elementwise_iou(A, C, "giou").item() == pytest.approx(-1 / 3, abs=1e-6)


def test_ciou_same_aspect_reduces_to_distance_term():
    # centres (5,5) and (10,10): rho2 = 50; enclosing diag^2 = 15^2 + 15^2 = 450
    # aspect term v = 0 because both are squares -> ciou = 1/7 - 50/450
    expected = 1 / 7 - 50 / 450
    assert elementwise_iou(A, B, "ciou").item() == pytest.approx(expected, abs=1e-6)


def test_ciou_disjoint():
    # centres (5,5),(25,5): rho2 = 400; enclosing 30x10 -> c2 = 1000; v = 0
    assert elementwise_iou(A, C, "ciou").item() == pytest.approx(-0.4, abs=1e-6)


def test_ciou_aspect_penalty():
    # A: 10x10 centre (5,5). D: 20x5 centre (10,2.5).
    # inter = [0,0,10,5] = 50; union = 100 + 100 - 50 = 150; iou = 1/3
    # enclosing [0,0,20,10]: c2 = 400 + 100 = 500; rho2 = 25 + 6.25 = 31.25
    # v = 4/pi^2 * (atan(20/5) - atan(10/10))^2
    # alpha = v / (v - iou + 1)      (gradient-free weight)
    iou = 1 / 3
    v = 4 / math.pi**2 * (math.atan(4.0) - math.atan(1.0)) ** 2
    alpha = v / (v - iou + 1)
    expected = iou - 31.25 / 500 - alpha * v
    assert elementwise_iou(A, D, "ciou").item() == pytest.approx(expected, abs=1e-5)


def test_iou_variants_are_symmetric():
    for kind in ("iou", "giou", "ciou"):
        ab = elementwise_iou(A, D, kind)
        ba = elementwise_iou(D, A, kind)
        assert ab.item() == pytest.approx(ba.item(), abs=1e-6)


def test_box_iou_matrix_shape_and_values():
    rows = torch.cat([A, C])  # (2, 4)
    cols = torch.cat([A, B, C])  # (3, 4)
    m = box_iou(rows, cols)
    assert m.shape == (2, 3)
    expected = torch.tensor([[1.0, 1 / 7, 0.0], [0.0, 0.0, 1.0]])
    assert torch.allclose(m, expected, atol=1e-6)


def test_ciou_gradient_flows_to_prediction():
    pred = B.clone().requires_grad_(True)
    loss = (1 - elementwise_iou(pred, A, "ciou")).sum()
    loss.backward()
    assert pred.grad is not None
    assert torch.isfinite(pred.grad).all()
    # Moving B towards A (negative x1/y1 direction) should reduce the loss.
    assert pred.grad[0, 0] > 0 and pred.grad[0, 1] > 0


def test_xyxy_cxcywh_roundtrip():
    boxes = torch.tensor([[0.0, 0.0, 10.0, 4.0], [3.0, 5.0, 9.0, 11.0]])
    cxcywh = xyxy_to_cxcywh(boxes)
    assert torch.equal(cxcywh, torch.tensor([[5.0, 2.0, 10.0, 4.0], [6.0, 8.0, 6.0, 6.0]]))
    assert torch.allclose(cxcywh_to_xyxy(cxcywh), boxes)


def test_ltrb_roundtrip():
    anchors = torch.tensor([[4.0, 4.0], [12.0, 20.0]])
    boxes = torch.tensor([[0.0, 0.0, 8.0, 8.0], [10.0, 14.0, 20.0, 30.0]])
    ltrb = xyxy_to_ltrb(anchors, boxes)
    assert torch.equal(ltrb, torch.tensor([[4.0, 4.0, 4.0, 4.0], [2.0, 6.0, 8.0, 10.0]]))
    assert torch.allclose(ltrb_to_xyxy(anchors, ltrb), boxes)
