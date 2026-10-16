from pathlib import Path

import pytest
import torch

from yolali.nn.config import ModelConfig
from yolali.nn.model import YolaliModel, build_model

CONFIGS = Path(__file__).resolve().parents[2] / "configs" / "models"

# Official gelan-c.pt has 25,498,752 parameters including the frozen 16-value
# DFL "conv"; yolali computes that arange instead of storing it.
GELAN_C_PARAMS = 25_498_752 - 16


def n_params(m: torch.nn.Module) -> int:
    return sum(p.numel() for p in m.parameters())


@pytest.fixture(scope="module")
def gelan_n() -> YolaliModel:
    return build_model(ModelConfig.from_yaml(CONFIGS / "gelan-n.yaml")).eval()


def test_gelan_c_parameter_count_matches_reference():
    model = build_model(ModelConfig.from_yaml(CONFIGS / "gelan-c.yaml"))
    assert n_params(model) == GELAN_C_PARAMS


def test_gelan_c_channels():
    model = build_model(ModelConfig.from_yaml(CONFIGS / "gelan-c.yaml"))
    assert model.backbone.out_channels == (512, 512, 512)
    assert model.neck.out_channels == (256, 512, 512)


def test_gelan_n_forward_shapes(gelan_n: YolaliModel):
    with torch.no_grad():
        pred = gelan_n(torch.zeros(2, 3, 320, 320))
    anchors = 40 * 40 + 20 * 20 + 10 * 10
    assert pred.box_raw.shape == (2, anchors, 64)
    assert pred.cls_logits.shape == (2, anchors, 80)
    assert pred.anchors.shape == (anchors, 2)
    assert pred.strides.shape == (anchors, 1)
    assert pred.strides[0].item() == 8 and pred.strides[-1].item() == 32


def test_gelan_n_decode_shapes(gelan_n: YolaliModel):
    with torch.no_grad():
        boxes, scores = gelan_n.decode(gelan_n(torch.zeros(1, 3, 320, 320)))
    assert boxes.shape == (1, 2100, 4)
    assert scores.shape == (1, 2100, 80)
    assert ((scores >= 0) & (scores <= 1)).all()


def test_rectangular_input(gelan_n: YolaliModel):
    with torch.no_grad():
        pred = gelan_n(torch.zeros(1, 3, 256, 320))
    assert pred.box_raw.shape[1] == 32 * 40 + 16 * 20 + 8 * 10


def test_fuse_preserves_outputs(gelan_n: YolaliModel):
    torch.manual_seed(0)
    model = build_model(ModelConfig.from_yaml(CONFIGS / "gelan-n.yaml")).eval()
    for bn in model.modules():
        if isinstance(bn, torch.nn.BatchNorm2d):
            with torch.no_grad():
                bn.running_mean.uniform_(-0.5, 0.5)  # type: ignore[union-attr]  # buffers exist on BatchNorm2d
                bn.running_var.uniform_(0.5, 1.5)  # type: ignore[union-attr]  # as above
                bn.weight.uniform_(0.5, 1.5)
    x = torch.randn(1, 3, 128, 128)
    with torch.no_grad():
        before = model(x)
        model.fuse()
        after = model(x)
    assert torch.allclose(before.box_raw, after.box_raw, atol=1e-4)
    assert torch.allclose(before.cls_logits, after.cls_logits, atol=1e-4)


def test_unknown_component_type_is_a_readable_error():
    cfg = ModelConfig.from_mapping(
        {
            "name": "x",
            "nc": 3,
            "backbone": {"type": "nope"},
            "neck": {"type": "gelan_pan"},
            "head": {"type": "detect"},
        }
    )
    with pytest.raises(ValueError, match="unknown component type"):
        build_model(cfg)


def test_config_requires_type_key():
    with pytest.raises(ValueError, match="needs a 'type' key"):
        ModelConfig.from_mapping(
            {"name": "x", "nc": 3, "backbone": {}, "neck": {"type": "a"}, "head": {"type": "b"}}
        )
