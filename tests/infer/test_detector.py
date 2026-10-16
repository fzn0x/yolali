from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from yolali.infer import Detections, Detector, draw_detections
from yolali.infer.preprocess import prepare_images, to_rgb_array
from yolali.nn.config import ModelConfig
from yolali.nn.model import YolaliModel, build_model


def tiny(head: str = "dual") -> YolaliModel:
    torch.manual_seed(0)
    reg_max = 0 if head == "dual" else 16
    cfg = ModelConfig.from_mapping(
        {
            "name": "tiny",
            "nc": 3,
            "backbone": {"type": "gelan", "width": 0.125, "depth": 0.33},
            "neck": {"type": "gelan_pan", "width": 0.125, "depth": 0.33},
            "head": {"type": head, "reg_max": reg_max},
        }
    )
    return build_model(cfg)


def test_to_rgb_array_accepts_every_input_kind(tmp_path: Path):
    arr = np.random.default_rng(0).integers(0, 255, (30, 40, 3), dtype=np.uint8)
    path = tmp_path / "img.png"
    Image.fromarray(arr).save(path)
    assert np.array_equal(to_rgb_array(path), arr)
    assert np.array_equal(to_rgb_array(Image.fromarray(arr)), arr)
    assert np.array_equal(to_rgb_array(arr), arr)
    t = torch.from_numpy(arr).permute(2, 0, 1).float() / 255
    assert np.abs(to_rgb_array(t).astype(int) - arr.astype(int)).max() <= 1
    gray = to_rgb_array(arr[..., 0])
    assert gray.shape == (30, 40, 3)


def test_prepare_images_letterboxes_to_square():
    arr = np.zeros((30, 60, 3), dtype=np.uint8)
    batch, metas = prepare_images([arr, arr], size=64)
    assert batch.shape == (2, 3, 64, 64) and batch.dtype == torch.float32
    assert metas[0].orig_hw == (30, 60)
    assert metas[0].scale == pytest.approx(64 / 60)
    assert metas[0].pad[1] == pytest.approx(16.0)  # 30 * 64/60 = 32 high -> pad 16 top


@pytest.mark.parametrize("head", ["dual", "detect"])
def test_predict_returns_one_detections_per_input(head: str, tmp_path: Path):
    det = Detector(tiny(head), device="cpu", names=["a", "b", "c"])
    arr = np.random.default_rng(1).integers(0, 255, (48, 80, 3), dtype=np.uint8)
    path = tmp_path / "x.jpg"
    Image.fromarray(arr).save(path)
    results = det.predict([arr, path, torch.from_numpy(arr).permute(2, 0, 1)], conf=0.0, size=64)
    assert len(results) == 3
    for r in results:
        assert isinstance(r, Detections)
        assert r.orig_hw == (48, 80)
        assert r.names == ("a", "b", "c")
        if len(r):
            assert (r.xyxy[:, 0] >= 0).all() and (r.xyxy[:, 2] <= 80).all()
            assert (r.xyxy[:, 1] >= 0).all() and (r.xyxy[:, 3] <= 48).all()
            assert (r.conf >= 0).all()
    single = det.predict(arr, conf=0.0, size=64)
    assert len(single) == 1


def test_detections_helpers():
    d = Detections(
        torch.tensor([[0.0, 0.0, 1.0, 1.0], [1.0, 1.0, 2.0, 2.0]]),
        torch.tensor([0.9, 0.2]),
        torch.tensor([1, 0]),
        names=("cat", "dog"),
    )
    assert len(d) == 2 and d.label(0) == "dog" and d.label(1) == "cat"
    assert len(d.filter(0.5)) == 1
    assert next(iter(d))[1] == pytest.approx(0.9)
    assert len(Detections.empty()) == 0
    with pytest.raises(ValueError, match="xyxy"):
        Detections(torch.zeros(2, 3), torch.zeros(2), torch.zeros(2, dtype=torch.int64))


def test_draw_detections_keeps_shape():
    img = np.zeros((40, 50, 3), dtype=np.uint8)
    d = Detections(
        torch.tensor([[5.0, 5.0, 30.0, 30.0]]), torch.tensor([0.7]), torch.tensor([0]), ("a",)
    )
    out = draw_detections(img, d)
    assert out.shape == img.shape and out.dtype == np.uint8
    assert out.sum() > 0 and img.sum() == 0  # drew on a copy
