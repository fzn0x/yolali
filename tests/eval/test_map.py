"""Hand-computed AP and matching cases."""

import numpy as np
import pytest
import torch

from yolali.eval import IOU_THRESHOLDS, MetricAccumulator, average_precision, match_predictions
from yolali.eval.evaluator import postprocess


def test_ap_perfect_single_prediction():
    assert average_precision(np.array([True]), np.array([0.9]), n_gt=1) == pytest.approx(1.0)


def test_ap_three_predictions_two_gts():
    # Ranked by score: TP, FP, TP -> precision 1, .5, .667 at recall .5, .5, 1.
    # Envelope from the right: [1, .667, .667]. Recall points 0..0.5 (51 of
    # them) sample precision 1; 0.51..1 (50) sample .667.
    tp = np.array([True, False, True])
    scores = np.array([0.9, 0.8, 0.7])
    expected = (51 * 1.0 + 50 * (2 / 3)) / 101
    assert average_precision(tp, scores, n_gt=2) == pytest.approx(expected, abs=1e-9)


def test_ap_missed_gt_caps_recall():
    # One TP, two GTs: recall never exceeds .5 -> precision 1 up to r=.5, 0 after.
    assert average_precision(np.array([True]), np.array([0.9]), n_gt=2) == pytest.approx(51 / 101)


def test_ap_no_gt_is_undefined_and_no_preds_is_zero():
    assert average_precision(np.zeros(0, dtype=bool), np.zeros(0), n_gt=0) == -1.0
    assert average_precision(np.zeros(0, dtype=bool), np.zeros(0), n_gt=3) == 0.0


def test_matching_greedy_by_score_and_class():
    gt = torch.tensor([[0.0, 0.0, 10.0, 10.0], [20.0, 20.0, 30.0, 30.0]])
    gt_cls = torch.tensor([0, 1])
    pred = torch.tensor(
        [
            [0.0, 0.0, 10.0, 10.0],  # exact match GT0, cls 0, score .9 -> TP everywhere
            [1.0, 1.0, 10.0, 10.0],  # duplicate of GT0 (IoU .81), lower score -> FP
            [20.0, 20.0, 30.0, 30.0],  # exact box but wrong class -> FP
            [21.0, 21.0, 30.0, 30.0],  # GT1 cls 1, IoU .81 -> TP for thr <= .8
        ]
    )
    scores = torch.tensor([0.9, 0.8, 0.7, 0.6])
    cls = torch.tensor([0, 0, 0, 1])
    tp, ignore = match_predictions(pred, scores, cls, gt, gt_cls)
    assert tp[0].all()
    assert not tp[1].any()
    assert not tp[2].any()
    assert tp[3].tolist() == (IOU_THRESHOLDS <= 0.81).tolist()
    assert not ignore.any()


def test_crowd_region_absorbs_predictions_without_tp():
    gt = torch.tensor([[0.0, 0.0, 100.0, 100.0]])
    pred = torch.tensor([[10.0, 10.0, 20.0, 20.0], [30.0, 30.0, 40.0, 40.0]])  # both inside
    tp, ignore = match_predictions(
        pred,
        torch.tensor([0.5, 0.4]),
        torch.tensor([0, 0]),
        gt,
        torch.tensor([0]),
        gt_crowd=torch.tensor([True]),
    )
    assert not tp.any() and ignore.all()


def test_regular_match_is_preferred_over_ignored():
    # GT0 regular (IoU .81 with the prediction), GT1 ignored (exact match).
    gt = torch.tensor([[1.0, 1.0, 10.0, 10.0], [0.0, 0.0, 10.0, 10.0]])
    pred = torch.tensor([[0.0, 0.0, 10.0, 10.0]])
    tp, ignore = match_predictions(
        pred,
        torch.tensor([0.9]),
        torch.tensor([0]),
        gt,
        torch.tensor([0, 0]),
        gt_ignore=torch.tensor([False, True]),
    )
    # TP while .81 clears the threshold, then falls back to the ignored GT.
    assert tp[0].tolist() == (IOU_THRESHOLDS <= 0.81).tolist()
    assert ignore[0].tolist() == (IOU_THRESHOLDS > 0.81).tolist()


def test_accumulator_end_to_end_hand_case():
    # One class, two images. Image 1: GT0 matched exactly, GT1 missed.
    # Image 2: one false positive. Ranked: TP (.9), FP (.8); n_gt = 2.
    # precision 1, .5 at recall .5, .5 -> AP = 51/101 at every threshold.
    acc = MetricAccumulator(num_classes=1)
    acc.add_image(
        torch.tensor([[0.0, 0.0, 50.0, 50.0]]),
        torch.tensor([0.9]),
        torch.tensor([0]),
        torch.tensor([[0.0, 0.0, 50.0, 50.0], [60.0, 60.0, 100.0, 100.0]]),
        torch.tensor([0, 0]),
    )
    acc.add_image(
        torch.tensor([[0.0, 0.0, 50.0, 50.0]]),
        torch.tensor([0.8]),
        torch.tensor([0]),
        torch.zeros(0, 4),
        torch.zeros(0, dtype=torch.int64),
    )
    m = acc.compute()
    assert m.map50 == pytest.approx(51 / 101)
    assert m.map50_95 == pytest.approx(51 / 101)
    # Both GTs are "medium" (2500 and 1600 px² lie in [32², 96²]).
    assert m.map_medium == pytest.approx(51 / 101)
    assert m.map_small == -1.0 and m.map_large == -1.0
    assert m.per_class_ap50_95 == {0: pytest.approx(51 / 101)}


def test_accumulator_caps_detections_per_class():
    acc = MetricAccumulator(num_classes=1, max_det=1)
    acc.add_image(
        torch.tensor([[0.0, 0.0, 50.0, 50.0], [0.0, 0.0, 50.0, 50.0]]),
        torch.tensor([0.5, 0.9]),
        torch.tensor([0, 0]),
        torch.tensor([[0.0, 0.0, 50.0, 50.0]]),
        torch.tensor([0]),
    )
    assert acc.compute().map50 == pytest.approx(1.0)  # only the .9 survives -> perfect


def test_postprocess_multilabel_and_nms():
    boxes = torch.tensor(
        [[[0.0, 0.0, 10.0, 10.0], [0.0, 0.0, 10.0, 10.0], [50.0, 50.0, 60.0, 60.0]]]
    )
    scores = torch.tensor([[[0.9, 0.6], [0.8, 0.1], [0.3, 0.7]]])
    ((_, s, c),) = postprocess(boxes, scores, conf=0.25, iou=0.5, max_det=10, nms=True)
    # Anchor 0 cls0 .9 / cls1 .6 pass; anchor 1 cls0 .8 is suppressed by anchor 0;
    # anchor 2 cls0 .3 / cls1 .7 pass (different box).
    assert s.tolist() == pytest.approx([0.9, 0.7, 0.6, 0.3])
    assert c.tolist() == [0, 1, 1, 0]
    ((_, s2, _),) = postprocess(boxes, scores, conf=0.25, iou=0.5, max_det=3, nms=False)
    assert s2.tolist() == pytest.approx([0.9, 0.8, 0.7])
