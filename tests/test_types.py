import pytest
import torch

from yolali.types import Targets


def test_from_list_pads_and_masks():
    boxes = [torch.tensor([[0.0, 0.0, 4.0, 4.0], [1.0, 1.0, 2.0, 2.0]]), torch.zeros(0, 4)]
    classes = [torch.tensor([3, 7]), torch.zeros(0, dtype=torch.int64)]
    t = Targets.from_list(boxes, classes)
    assert t.batch_size == 2 and t.max_boxes == 2
    assert t.mask.tolist() == [[True, True], [False, False]]
    assert t.classes.tolist() == [[3, 7], [0, 0]]
    assert torch.equal(t.boxes[1], torch.zeros(2, 4))


def test_all_empty_batch_has_width_one():
    t = Targets.from_list([torch.zeros(0, 4)], [torch.zeros(0, dtype=torch.int64)])
    assert t.boxes.shape == (1, 1, 4)
    assert not t.mask.any()


def test_shape_validation():
    with pytest.raises(ValueError, match="boxes must be"):
        Targets(torch.zeros(2, 4), torch.zeros(2), torch.zeros(2, dtype=torch.bool))
