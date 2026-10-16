import torch

from yolali.ops import make_anchors


def test_two_level_toy_pyramid():
    # Level 0: 2x2 cells at stride 8 -> centres at 4 and 12.
    # Level 1: 1x1 cell at stride 16 -> centre at 8.
    anchors, strides = make_anchors([(2, 2), (1, 1)], [8, 16])
    expected = torch.tensor([[4.0, 4.0], [12.0, 4.0], [4.0, 12.0], [12.0, 12.0], [8.0, 8.0]])
    assert torch.equal(anchors, expected)
    assert torch.equal(strides, torch.tensor([[8.0], [8.0], [8.0], [8.0], [16.0]]))


def test_row_major_order_for_non_square_level():
    # 2 rows x 3 cols at stride 4: x varies fastest.
    anchors, _ = make_anchors([(2, 3)], [4])
    assert torch.equal(anchors[:, 0], torch.tensor([2.0, 6.0, 10.0, 2.0, 6.0, 10.0]))
    assert torch.equal(anchors[:, 1], torch.tensor([2.0, 2.0, 2.0, 6.0, 6.0, 6.0]))


def test_zero_offset_puts_anchors_on_cell_corners():
    anchors, _ = make_anchors([(1, 2)], [8], offset=0.0)
    assert torch.equal(anchors, torch.tensor([[0.0, 0.0], [8.0, 0.0]]))
