"""Converter mapping (always) and forward-pass equivalence (with official weights).

``gelan_c_reference_shapes.json`` is the reference model's ``state_dict``
key -> shape table; ``*_320.npz`` hold the reference forward pass on a fixed
uint8 image, both produced by running the GPL reference code in a separate
checkout (see docs/plan.md phase 2). The equivalence tests additionally need
the official checkpoints in ``./weights`` and are skipped without them.
"""

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from yolali.nn.config import ModelConfig
from yolali.nn.model import YolaliModel, build_model
from yolali.ops import xyxy_to_cxcywh
from yolali.zoo import convert_gelan_c_state_dict, load_yolov9_gelan_c

ROOT = Path(__file__).resolve().parents[2]
FIXTURES = ROOT / "tests" / "fixtures"
WEIGHTS = ROOT / "weights"
CHECKPOINTS = ("gelan-c", "yolov9-c-converted")


def gelan_c() -> YolaliModel:
    return build_model(ModelConfig.from_yaml(ROOT / "configs" / "models" / "gelan-c.yaml")).eval()


def test_every_reference_key_maps_to_a_model_key_with_the_same_shape():
    shapes: dict[str, list[int]] = json.loads(
        (FIXTURES / "gelan_c_reference_shapes.json").read_text()
    )
    fake = {k: torch.empty(shape) for k, shape in shapes.items()}  # shape [] is a scalar
    converted = convert_gelan_c_state_dict(fake)
    ours = gelan_c().state_dict()
    assert set(converted) == set(ours), set(converted) ^ set(
        ours
    )  # symmetric difference shows exactly what is off
    for key, tensor in converted.items():
        assert tuple(tensor.shape) == tuple(ours[key].shape), key


def test_unmapped_key_is_an_error():
    with pytest.raises(ValueError, match="no mapping"):
        convert_gelan_c_state_dict({"model.99.conv.weight": torch.zeros(1)})


@pytest.mark.weights
@pytest.mark.parametrize("name", CHECKPOINTS)
def test_forward_equivalence_with_official_checkpoint(name: str):
    ckpt = WEIGHTS / f"{name}.pt"
    ref_path = FIXTURES / f"{name}_320.npz"
    if not ckpt.exists():
        pytest.skip(f"{ckpt} not present; download from the YOLOv9 releases")
    ref = np.load(ref_path)

    model = gelan_c()
    model.load_state_dict(load_yolov9_gelan_c(ckpt), strict=True)

    x = torch.from_numpy(ref["input_u8"]).float() / 255.0  # (1, 3, 320, 320)
    with torch.no_grad():
        pred = model(x)
        boxes, scores = model.decode(pred)

    # Reference raw outputs are per level (1, 64 + 80, H, W); ours are flattened (1, A, C).
    offset = 0
    for level in range(3):
        raw = torch.from_numpy(ref[f"raw{level}"])  # (1, 144, H, W)
        _, _, h, w = raw.shape
        n = h * w
        ref_box = raw[:, :64].reshape(1, 64, n).permute(0, 2, 1)
        ref_cls = raw[:, 64:].reshape(1, 80, n).permute(0, 2, 1)
        box_diff = (pred.box_raw[:, offset : offset + n] - ref_box).abs().max().item()
        cls_diff = (pred.cls_logits[:, offset : offset + n] - ref_cls).abs().max().item()
        assert box_diff < 1e-4, f"level {level} box max abs diff {box_diff}"
        assert cls_diff < 1e-4, f"level {level} cls max abs diff {cls_diff}"
        offset += n
    assert offset == pred.box_raw.shape[1]

    # Decoded output: reference y is (1, 4 + 80, A) with cxcywh pixels then sigmoid scores.
    y = torch.from_numpy(ref["y"])
    ours = torch.cat((xyxy_to_cxcywh(boxes), scores), 2).permute(0, 2, 1)
    decoded_diff = (ours - y).abs().max().item()
    assert decoded_diff < 1e-3, f"decoded max abs diff {decoded_diff}"
