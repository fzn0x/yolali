import torch

from yolali.loss import DualLoss, LossConfig
from yolali.nn.config import ModelConfig
from yolali.nn.head.detect import DetectHead
from yolali.nn.head.dual import DualDetectHead
from yolali.nn.model import build_model
from yolali.types import DualRawPrediction, RawPrediction, Targets

CH = (16, 32, 32)


def n_params(m: torch.nn.Module) -> int:
    return sum(p.numel() for p in m.parameters())


def feats(batch: int = 1) -> list[torch.Tensor]:
    return [torch.randn(batch, c, s, s) for c, s in zip(CH, (8, 4, 2), strict=True)]


def test_training_returns_both_branches_eval_returns_o2o():
    head = DualDetectHead(CH, num_classes=3, reg_max=0)
    out = head(feats())
    assert isinstance(out, DualRawPrediction)
    assert out.o2m.box_raw.shape == (1, 84, 4) and out.o2o.cls_logits.shape == (1, 84, 3)
    head.eval()
    with torch.no_grad():
        single = head(feats())
    assert isinstance(single, RawPrediction)
    # The eval output is exactly the one-to-one branch.
    torch.manual_seed(0)
    x = feats()
    with torch.no_grad():
        assert torch.equal(head(x).box_raw, head.o2o(x).box_raw)


def test_parameter_count_is_two_detect_heads():
    dual = DualDetectHead(CH, 3, reg_max=16)
    single = DetectHead(CH, 3, reg_max=16)
    assert n_params(dual) == 2 * n_params(single)


def test_o2o_branch_does_not_backpropagate_into_features():
    head = DualDetectHead(CH, 2, reg_max=0)
    x = [f.requires_grad_(True) for f in feats()]
    out = head(x)
    assert isinstance(out, DualRawPrediction)
    out.o2o.cls_logits.sum().backward()
    assert all(f.grad is None for f in x)
    out = head([f.detach().requires_grad_(True) for f in x])
    assert isinstance(out, DualRawPrediction)
    out.o2m.cls_logits.sum().backward()


def test_o2o_branch_can_be_attached():
    head = DualDetectHead(CH, 2, reg_max=0, detach_o2o=False)
    x = [f.requires_grad_(True) for f in feats()]
    out = head(x)
    assert isinstance(out, DualRawPrediction)
    out.o2o.cls_logits.sum().backward()
    assert all(f.grad is not None for f in x)


def test_model_with_dual_head_trains_and_decodes():
    cfg = ModelConfig.from_mapping(
        {
            "name": "dual-tiny",
            "nc": 2,
            "backbone": {"type": "gelan", "width": 0.125, "depth": 0.33},
            "neck": {"type": "gelan_pan", "width": 0.125, "depth": 0.33},
            "head": {"type": "dual", "reg_max": 0},
        }
    )
    model = build_model(cfg)
    assert model.reg_max == 0
    images = torch.rand(2, 3, 64, 64)
    targets = Targets.from_list(
        [torch.tensor([[8.0, 8.0, 40.0, 40.0]]), torch.zeros(0, 4)],
        [torch.tensor([1]), torch.zeros(0, dtype=torch.int64)],
    )
    out = model(images)
    assert isinstance(out, DualRawPrediction)
    loss = DualLoss(LossConfig(reg_max=0), 2)(out, targets)
    loss.total.backward()
    assert torch.isfinite(loss.total)
    model.eval()
    with torch.no_grad():
        pred = model(images)
        assert isinstance(pred, RawPrediction)
        boxes, scores = model.decode(pred)
    assert boxes.shape == (2, 84, 4) and scores.shape == (2, 84, 2)
