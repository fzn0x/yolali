"""End-to-end loss on single-anchor cases where every term has a closed form.

One level at stride 8 with a 1x1 feature map gives one anchor at (4, 4).
"""

import math

import pytest
import torch

from yolali.loss import DetectionLoss, DualLoss, LossConfig
from yolali.ops import make_anchors
from yolali.types import DualRawPrediction, RawPrediction, Targets

LN2 = math.log(2.0)
ANCHORS, STRIDES = make_anchors([(1, 1)], [8])
SOFTPLUS_INV_HALF = math.log(math.exp(0.5) - 1)  # softplus(x) = 0.5 -> 4 px at stride 8


def raw(box_raw: torch.Tensor, cls_logits: torch.Tensor) -> RawPrediction:
    return RawPrediction(box_raw, cls_logits, ANCHORS, STRIDES)


def one_box(xyxy: list[float], cls: int = 0) -> Targets:
    return Targets(torch.tensor([[xyxy]]), torch.tensor([[cls]]), torch.tensor([[True]]))


def test_direct_regression_exact_prediction_leaves_only_cls_term():
    # Prediction decodes to [0,0,8,8] == GT, so iou = 1 and box loss = 0.
    # Score p = .5 for the GT class -> metric is the max -> soft target = iou_max = 1.
    # cls = BCE(0, 1) = ln2; target mass = 1; total = 0.5 * ln2 * batch(1).
    cfg = LossConfig(reg_max=0)
    loss = DetectionLoss(cfg, num_classes=1)
    box_raw = torch.full((1, 1, 4), SOFTPLUS_INV_HALF)
    cls_logits = torch.zeros(1, 1, 1)
    out = loss(raw(box_raw, cls_logits), one_box([0.0, 0.0, 8.0, 8.0]))
    assert out.num_fg == 1
    assert out.box.item() == pytest.approx(0.0, abs=1e-5)
    assert out.cls.item() == pytest.approx(LN2, abs=1e-6)
    assert out.dfl.item() == 0.0
    assert out.total.item() == pytest.approx(0.5 * LN2, abs=1e-5)


def test_dfl_exact_expectation_costs_ln2_per_side():
    # Equal logits on bins 0 and 1 -> expectation 0.5 stride = 4 px -> box == GT.
    # DFL target 0.5 brackets bins 0 and 1 with weight .5 each: .5 ln2 + .5 ln2 = ln2.
    # total = 0.5 * ln2 (cls) + 1.5 * ln2 (dfl) = 2 ln2.
    reg_max = 16
    cfg = LossConfig(reg_max=reg_max)
    loss = DetectionLoss(cfg, num_classes=1)
    box_raw = torch.zeros(1, 1, 4 * reg_max)
    for side in range(4):
        box_raw[0, 0, side * reg_max] = 20.0
        box_raw[0, 0, side * reg_max + 1] = 20.0
    out = loss(raw(box_raw, torch.zeros(1, 1, 1)), one_box([0.0, 0.0, 8.0, 8.0]))
    assert out.box.item() == pytest.approx(0.0, abs=1e-4)
    assert out.cls.item() == pytest.approx(LN2, abs=1e-6)
    assert out.dfl.item() == pytest.approx(LN2, abs=1e-4)
    assert out.total.item() == pytest.approx(2 * LN2, abs=1e-4)


def test_direct_regression_undersized_prediction():
    # Pred [0,0,8,8] vs GT [0,0,16,16]: iou = 64/256 = .25, soft target = .25.
    # Target mass = max(.25, 1) = 1.
    # cls = BCE(0, .25) = ln2.
    # ciou: centres (4,4),(8,8) rho2 = 32; enclosing 16x16 c2 = 512; v = 0
    #       -> .25 - 32/512 = .1875; box = weight .25 * (1 - .1875) = .203125
    cfg = LossConfig(reg_max=0)
    loss = DetectionLoss(cfg, num_classes=1)
    box_raw = torch.full((1, 1, 4), SOFTPLUS_INV_HALF)
    out = loss(raw(box_raw, torch.zeros(1, 1, 1)), one_box([0.0, 0.0, 16.0, 16.0]))
    assert out.cls.item() == pytest.approx(LN2, abs=1e-6)
    assert out.box.item() == pytest.approx(0.203125, abs=1e-5)
    assert out.total.item() == pytest.approx(7.5 * 0.203125 + 0.5 * LN2, abs=1e-5)


def test_no_positives_gives_cls_only_and_stays_differentiable():
    # GT far outside the only anchor: nothing is assigned.
    cfg = LossConfig(reg_max=0)
    loss = DetectionLoss(cfg, num_classes=2)
    box_raw = torch.zeros(1, 1, 4, requires_grad=True)
    cls_logits = torch.zeros(1, 1, 2, requires_grad=True)
    out = loss(raw(box_raw, cls_logits), one_box([100.0, 100.0, 120.0, 120.0]))
    assert out.num_fg == 0
    assert out.box.item() == 0.0 and out.dfl.item() == 0.0
    # Two negatives at p = .5 -> 2 ln2, divided by mass clamp 1.
    assert out.cls.item() == pytest.approx(2 * LN2, abs=1e-6)
    out.total.backward()
    assert cls_logits.grad is not None and box_raw.grad is not None


def test_batch_size_scales_total():
    cfg = LossConfig(reg_max=0)
    loss = DetectionLoss(cfg, num_classes=1)
    single = loss(
        raw(torch.full((1, 1, 4), SOFTPLUS_INV_HALF), torch.zeros(1, 1, 1)),
        one_box([0.0, 0.0, 8.0, 8.0]),
    )
    targets = Targets.from_list([torch.tensor([[0.0, 0.0, 8.0, 8.0]])] * 2, [torch.tensor([0])] * 2)
    double = loss(raw(torch.full((2, 1, 4), SOFTPLUS_INV_HALF), torch.zeros(2, 1, 1)), targets)
    # Per-term values are averages over target mass (2 here), total multiplies by B.
    assert double.cls.item() == pytest.approx(single.cls.item(), abs=1e-6)
    assert double.total.item() == pytest.approx(2 * single.total.item(), abs=1e-5)


def test_varifocal_option_changes_only_cls():
    bce = DetectionLoss(LossConfig(reg_max=0, cls_loss="bce"), 1)
    vfl = DetectionLoss(LossConfig(reg_max=0, cls_loss="vfl"), 1)
    args = (
        raw(torch.full((1, 1, 4), SOFTPLUS_INV_HALF), torch.zeros(1, 1, 1)),
        one_box([0.0, 0.0, 8.0, 8.0]),
    )
    a, b = bce(*args), vfl(*args)
    assert a.box.item() == pytest.approx(b.box.item())
    # Positive with soft target 1: VFL weight = target = 1 -> same as BCE here.
    assert b.cls.item() == pytest.approx(LN2, abs=1e-6)


def test_dual_loss_combines_branches():
    cfg = LossConfig(reg_max=0, o2o_weight=0.5)
    loss = DualLoss(cfg, num_classes=1)
    branch = raw(torch.full((1, 1, 4), SOFTPLUS_INV_HALF), torch.zeros(1, 1, 1))
    out = loss(DualRawPrediction(branch, branch), one_box([0.0, 0.0, 8.0, 8.0]))
    assert out.o2m.total.item() == pytest.approx(0.5 * LN2, abs=1e-5)
    assert out.o2o.total.item() == pytest.approx(0.5 * LN2, abs=1e-5)
    assert out.total.item() == pytest.approx(0.5 * LN2 * 1.5, abs=1e-5)
    keys = set(out.as_dict())
    assert {"total", "o2m/box", "o2o/cls", "o2o/num_fg"} <= keys


def test_config_validation():
    with pytest.raises(ValueError):
        LossConfig(reg_max=-1)
    with pytest.raises(ValueError):
        LossConfig(iou_kind="iou")
