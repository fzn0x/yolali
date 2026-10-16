import torch

from yolali.ops import batched_nms

# Five boxes, two classes. IoU([0,0,10,10],[1,1,11,11]) = 81/(200-81) = 0.68.
BOXES = torch.tensor(
    [
        [0.0, 0.0, 10.0, 10.0],  # 0: cls 0, score .9  -> kept
        [1.0, 1.0, 11.0, 11.0],  # 1: cls 0, score .8  -> suppressed by 0
        [1.0, 1.0, 11.0, 11.0],  # 2: cls 1, score .7  -> kept (other class)
        [20.0, 20.0, 30.0, 30.0],  # 3: cls 0, score .6 -> kept (disjoint)
        [0.0, 0.0, 10.0, 10.0],  # 4: cls 1, score .5  -> suppressed by 2
    ]
)
SCORES = torch.tensor([0.9, 0.8, 0.7, 0.6, 0.5])
CLASSES = torch.tensor([0, 0, 1, 0, 1])


def test_keeps_expected_indices_in_score_order():
    keep = batched_nms(BOXES, SCORES, CLASSES, iou_threshold=0.5)
    assert keep.tolist() == [0, 2, 3]


def test_high_threshold_keeps_everything():
    keep = batched_nms(BOXES, SCORES, CLASSES, iou_threshold=0.7)
    assert keep.tolist() == [0, 1, 2, 3, 4]


def test_max_det_truncates_after_sorting():
    keep = batched_nms(BOXES, SCORES, CLASSES, iou_threshold=0.5, max_det=2)
    assert keep.tolist() == [0, 2]


def test_empty_input():
    keep = batched_nms(torch.zeros(0, 4), torch.zeros(0), torch.zeros(0, dtype=torch.int64), 0.5)
    assert keep.shape == (0,)
    assert keep.dtype == torch.int64
