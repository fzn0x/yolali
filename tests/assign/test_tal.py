"""Hand-worked task-aligned assignment on a 4x4 toy grid.

Image 32x32, one level at stride 8 -> 16 anchors at x, y in {4, 12, 20, 28}.
Anchor index = row * 4 + col, e.g. (12, 4) is index 1, (4, 12) is index 4.
Every predicted box is an 8x8 square centred on its anchor, so an anchor
inside a 16x16 ground truth has IoU 64 / 256 = 0.25 with it.
"""

import math

import pytest
import torch

from yolali.assign import OneToOneAssigner, TaskAlignedAssigner
from yolali.ops import ltrb_to_xyxy, make_anchors
from yolali.types import Targets

ANCHORS, _ = make_anchors([(4, 4)], [8])
PRED_BOXES = ltrb_to_xyxy(ANCHORS, torch.full((16, 4), 4.0)).unsqueeze(0)  # (1, 16, 4)
NC = 2


def scores_with(default: float = 0.5, **overrides: float) -> torch.Tensor:
    """(1, 16, NC) class probabilities; overrides like a5_c0=1.0 set anchor 5 class 0."""
    s = torch.full((1, 16, NC), default)
    for key, val in overrides.items():
        a, c = key.split("_")
        s[0, int(a[1:]), int(c[1:])] = val
    return s


def two_disjoint_boxes() -> Targets:
    # GT0 = [0,0,16,16] cls 0 contains anchors 0,1,4,5. GT1 = [16,16,32,32] cls 1: 10,11,14,15.
    return Targets(
        boxes=torch.tensor([[[0.0, 0.0, 16.0, 16.0], [16.0, 16.0, 32.0, 32.0]]]),
        classes=torch.tensor([[0, 1]]),
        mask=torch.tensor([[True, True]]),
    )


def test_all_inside_anchors_are_positive_when_topk_exceeds_them():
    assigner = TaskAlignedAssigner(topk=10)
    out = assigner(scores_with(a5_c0=1.0), PRED_BOXES, ANCHORS, two_disjoint_boxes())
    expected_fg = torch.zeros(16, dtype=torch.bool)
    expected_fg[[0, 1, 4, 5, 10, 11, 14, 15]] = True
    assert torch.equal(out.fg_mask[0], expected_fg)
    assert out.gt_idx[0, [0, 1, 4, 5]].tolist() == [0, 0, 0, 0]
    assert out.gt_idx[0, [10, 11, 14, 15]].tolist() == [1, 1, 1, 1]
    # Each positive anchor regresses to its box.
    assert torch.equal(out.boxes[0, 5], torch.tensor([0.0, 0.0, 16.0, 16.0]))
    assert torch.equal(out.boxes[0, 10], torch.tensor([16.0, 16.0, 32.0, 32.0]))
    # Background anchors have zero box and zero scores.
    assert torch.equal(out.boxes[0, 2], torch.zeros(4))
    assert torch.equal(out.scores[0, 2], torch.zeros(NC))


def test_normalised_soft_scores():
    # GT0: metric_i = s_i^0.5 * 0.25^6. Anchor 5 has s = 1 -> the max.
    # Soft target = metric_i / metric_max * iou_max = sqrt(s_i) * 0.25.
    out = TaskAlignedAssigner(topk=10)(
        scores_with(a5_c0=1.0), PRED_BOXES, ANCHORS, two_disjoint_boxes()
    )
    assert out.scores[0, 5, 0].item() == pytest.approx(0.25, abs=1e-6)
    assert out.scores[0, 0, 0].item() == pytest.approx(math.sqrt(0.5) * 0.25, abs=1e-6)
    # Wrong-class column is zero for positives.
    assert out.scores[0, 5, 1].item() == 0.0
    # GT1 anchors all share s = 0.5 -> each gets the full iou_max = 0.25.
    assert out.scores[0, 10, 1].item() == pytest.approx(0.25, abs=1e-6)


def test_topk_limits_candidates_by_metric():
    # topk=2 on GT0: the two highest-scoring inside anchors win.
    out = TaskAlignedAssigner(topk=2)(
        scores_with(a5_c0=1.0, a4_c0=0.9), PRED_BOXES, ANCHORS, two_disjoint_boxes()
    )
    gt0_positives = out.fg_mask[0, [0, 1, 4, 5]].tolist()
    assert gt0_positives == [False, False, True, True]


def test_conflict_resolved_by_iou():
    # GT0 = [0,0,16,16], GT1 = [8,8,24,24]; anchor 5 at (12,12) lies in both.
    # Make anchor 5 predict [8,8,18,18] (area 100):
    #   IoU with GT0: inter [8,8,16,16] = 64, union 256+100-64 = 292 -> 0.219
    #   IoU with GT1: inter 100, union 256 -> 0.391  => GT1 wins.
    targets = Targets(
        boxes=torch.tensor([[[0.0, 0.0, 16.0, 16.0], [8.0, 8.0, 24.0, 24.0]]]),
        classes=torch.tensor([[0, 0]]),
        mask=torch.tensor([[True, True]]),
    )
    pred = PRED_BOXES.clone()
    pred[0, 5] = torch.tensor([8.0, 8.0, 18.0, 18.0])
    out = TaskAlignedAssigner(topk=10)(scores_with(), pred, ANCHORS, targets)
    assert bool(out.fg_mask[0, 5])
    assert out.gt_idx[0, 5].item() == 1
    assert torch.equal(out.boxes[0, 5], torch.tensor([8.0, 8.0, 24.0, 24.0]))
    # Anchors 0, 1, 4 are only in GT0 and stay with it.
    assert out.gt_idx[0, [0, 1, 4]].tolist() == [0, 0, 0]


def test_padded_ground_truth_gets_nothing():
    targets = Targets(
        boxes=torch.tensor([[[0.0, 0.0, 16.0, 16.0], [0.0, 0.0, 0.0, 0.0]]]),
        classes=torch.tensor([[0, 0]]),
        mask=torch.tensor([[True, False]]),
    )
    out = TaskAlignedAssigner(topk=10)(scores_with(), PRED_BOXES, ANCHORS, targets)
    assert out.fg_mask[0].sum().item() == 4
    assert (out.gt_idx[0][out.fg_mask[0]] == 0).all()


def test_empty_image_in_batch():
    targets = Targets.from_list(
        [torch.tensor([[0.0, 0.0, 16.0, 16.0]]), torch.zeros(0, 4)],
        [torch.tensor([0]), torch.zeros(0, dtype=torch.int64)],
    )
    scores = scores_with().expand(2, -1, -1)
    pred = PRED_BOXES.expand(2, -1, -1)
    out = TaskAlignedAssigner(topk=10)(scores, pred, ANCHORS, targets)
    assert out.fg_mask[0].sum().item() == 4
    assert out.fg_mask[1].sum().item() == 0
    assert out.scores[1].sum().item() == 0.0


def test_one_to_one_picks_single_best_anchor_per_box():
    # GT0's best metric is anchor 5 (s=1.0); GT1's best is anchor 11 (s=0.9).
    out = OneToOneAssigner()(
        scores_with(a5_c0=1.0, a11_c1=0.9), PRED_BOXES, ANCHORS, two_disjoint_boxes()
    )
    assert out.fg_mask[0].nonzero().flatten().tolist() == [5, 11]
    assert out.gt_idx[0, 5].item() == 0 and out.gt_idx[0, 11].item() == 1
    # Its own metric is the max, so the soft target equals iou_max = 0.25.
    assert out.scores[0, 5, 0].item() == pytest.approx(0.25, abs=1e-6)
    assert out.scores[0, 11, 1].item() == pytest.approx(0.25, abs=1e-6)


def test_one_to_one_never_assigns_two_anchors_to_one_box():
    targets = two_disjoint_boxes()
    out = OneToOneAssigner()(scores_with(), PRED_BOXES, ANCHORS, targets)
    for m in range(targets.max_boxes):
        assert int(((out.gt_idx[0] == m) & out.fg_mask[0]).sum()) <= 1
