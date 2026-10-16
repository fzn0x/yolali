from typing import Any, cast

import pytest
import torch

from yolali.nn.backbone.sta import SpatialTuningAdapter
from yolali.nn.config import ModelConfig
from yolali.nn.model import build_model

timm = pytest.importorskip("timm")
from yolali.nn.backbone.vit import ViTBackbone  # noqa: E402 - after importorskip


def test_sta_shapes_and_channels():
    sta = SpatialTuningAdapter(96, (32, 64, 96))
    p3, p4, p5 = sta(torch.zeros(2, 96, 20, 20))
    assert p3.shape == (2, 32, 40, 40)
    assert p4.shape == (2, 64, 20, 20)
    assert p5.shape == (2, 96, 10, 10)


@pytest.fixture(scope="module")
def vit_model() -> ViTBackbone:
    torch.manual_seed(0)
    return ViTBackbone(
        "vit_small_patch14_dinov2", (32, 64, 96), img_size=128, pretrained=False, freeze=False
    )


@pytest.mark.parametrize("size", [(128, 128), (96, 160)])
def test_vit_backbone_output_strides(vit_model: ViTBackbone, size: tuple[int, int]):
    with torch.no_grad():
        p3, p4, p5 = vit_model(torch.zeros(1, 3, *size))
    h, w = size
    assert p3.shape == (1, 32, h // 8, w // 8)
    assert p4.shape == (1, 64, h // 16, w // 16)
    assert p5.shape == (1, 96, h // 32, w // 32)


def test_vit_rejects_non_multiple_of_32(vit_model: ViTBackbone):
    with pytest.raises(ValueError, match="multiples of 32"):
        vit_model(torch.zeros(1, 3, 100, 128))


def test_patch_size_override_resamples_embedding(vit_model: ViTBackbone):
    patch_embed = cast(Any, vit_model.vit).patch_embed
    assert tuple(patch_embed.patch_size) == (16, 16)


def test_freeze_leaves_vit_without_gradients():
    bb = ViTBackbone(
        "vit_small_patch14_dinov2", (16, 32, 48), img_size=64, pretrained=False, freeze=True
    )
    bb.train()
    assert not bb.vit.training and bb.adapter.training
    out = bb(torch.rand(1, 3, 64, 64))
    torch.stack([o.sum() for o in out]).sum().backward()
    assert all(p.grad is None for p in bb.vit.parameters())
    assert any(p.grad is not None for p in bb.adapter.parameters())


def test_vit_model_from_config_end_to_end():
    cfg = ModelConfig.from_mapping(
        {
            "name": "vit-tiny",
            "nc": 4,
            "backbone": {
                "type": "vit",
                "model": "vit_small_patch14_dinov2",
                "out_channels": [16, 32, 48],
                "img_size": 64,
                "pretrained": False,
            },
            "neck": {"type": "gelan_pan", "width": 0.125, "depth": 0.33},
            "head": {"type": "dual", "reg_max": 0},
        }
    )
    model = build_model(cfg).eval()
    with torch.no_grad():
        pred = model(torch.rand(1, 3, 64, 64))
        boxes, scores = model.decode(pred)
    assert boxes.shape == (1, 64 + 16 + 4, 4) and scores.shape[-1] == 4


@pytest.mark.slow
def test_pretrained_dinov2_small_downloads_and_runs():
    bb = ViTBackbone(
        "vit_small_patch14_dinov2.lvd142m", (64, 128, 192), img_size=224, pretrained=True
    ).eval()
    with torch.no_grad():
        p3, _, _ = bb(torch.rand(1, 3, 224, 224))
    assert p3.shape == (1, 64, 28, 28)
