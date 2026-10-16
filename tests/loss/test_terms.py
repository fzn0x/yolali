"""Hand-computed values for the individual loss terms."""

import math

import pytest
import torch

from yolali.loss import bce_loss, dfl_loss, iou_loss, varifocal_loss

LN2 = math.log(2.0)


def test_bce_on_zero_logits_is_ln2_per_element():
    logits = torch.zeros(1, 2)
    targets = torch.tensor([[1.0, 0.0]])
    assert bce_loss(logits, targets).item() == pytest.approx(2 * LN2, abs=1e-6)


def test_bce_soft_target_at_half_probability():
    # -[t ln .5 + (1-t) ln .5] = ln 2 regardless of t.
    assert bce_loss(torch.zeros(1, 1), torch.tensor([[0.25]])).item() == pytest.approx(LN2)


def test_varifocal_weights():
    # Positive (target 1): weight = target = 1 -> ln2.
    # Negative (target 0, p = .5): weight = .75 * .5^2 = .1875 -> .1875 ln2.
    logits = torch.zeros(1, 2)
    targets = torch.tensor([[1.0, 0.0]])
    expected = LN2 + 0.1875 * LN2
    assert varifocal_loss(logits, targets, 0.75, 2.0).item() == pytest.approx(expected, abs=1e-6)


def test_iou_loss_is_weighted_one_minus_ciou():
    a = torch.tensor([[0.0, 0.0, 10.0, 10.0]])
    b = torch.tensor([[5.0, 5.0, 15.0, 15.0]])
    ciou = 1 / 7 - 50 / 450  # see tests/ops/test_boxes.py
    out = iou_loss(a, b, torch.tensor([2.0]), "ciou")
    assert out.item() == pytest.approx(2 * (1 - ciou), abs=1e-6)


def test_dfl_uniform_logits_cost_log_reg_max():
    # Uniform distribution over 4 bins: CE = ln 4 for either bracket bin,
    # weights sum to 1 -> ln 4 per side, mean over sides = ln 4, times weight 3.
    pred = torch.zeros(1, 16)
    target = torch.full((1, 4), 2.5)
    out = dfl_loss(pred, target, torch.tensor([3.0]), reg_max=4)
    assert out.item() == pytest.approx(3 * math.log(4), abs=1e-6)


def test_dfl_exact_integer_target_with_confident_logits_is_zero():
    reg_max = 4
    pred = torch.full((1, 4 * reg_max), -1e4)
    for side in range(4):
        pred[0, side * reg_max + 2] = 0.0  # all mass on bin 2
    out = dfl_loss(pred, torch.full((1, 4), 2.0), torch.tensor([1.0]), reg_max)
    assert out.item() == pytest.approx(0.0, abs=1e-6)


def test_dfl_clamps_target_to_last_bracket():
    # Target beyond the last bin is clamped to reg_max - 1.01, so no index error.
    out = dfl_loss(torch.zeros(1, 16), torch.full((1, 4), 99.0), torch.tensor([1.0]), 4)
    assert torch.isfinite(out)


def test_dfl_empty_is_zero_and_connected():
    pred = torch.zeros(0, 16, requires_grad=True)
    out = dfl_loss(pred, torch.zeros(0, 4), torch.zeros(0), 4)
    assert out.item() == 0.0
    out.backward()
    assert pred.grad is not None
