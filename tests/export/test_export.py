"""Exported graphs must reproduce PyTorch outputs; the NMS-free graph must use only standard ops."""

from pathlib import Path
from typing import cast

import numpy as np
import pytest
import torch

from yolali.export import ExportWrapper, export_onnx, export_torchscript
from yolali.nn.config import ModelConfig
from yolali.nn.model import YolaliModel, build_model

SIZE = 96


def tiny(head: str) -> YolaliModel:
    torch.manual_seed(0)
    cfg = ModelConfig.from_mapping(
        {
            "name": "tiny",
            "nc": 3,
            "backbone": {"type": "gelan", "width": 0.125, "depth": 0.33},
            "neck": {"type": "gelan_pan", "width": 0.125, "depth": 0.33},
            "head": {"type": head, "reg_max": 0 if head == "dual" else 16},
        }
    )
    model = build_model(cfg).eval()
    # Non-trivial BatchNorm statistics so fusion and export are exercised honestly.
    for bn in model.modules():
        if isinstance(bn, torch.nn.BatchNorm2d):
            state = bn.state_dict()
            state["running_mean"].uniform_(-0.5, 0.5)
            state["running_var"].uniform_(0.5, 1.5)
    return model


def test_wrapper_nms_free_output_layout():
    model = tiny("dual")
    out = ExportWrapper(model, max_det=10)(torch.rand(2, 3, SIZE, SIZE))
    assert out.shape == (2, 10, 6)
    assert (out[..., 4] >= 0).all() and (out[..., 4] <= 1).all()
    # Sorted by confidence, classes are integral.
    assert (out[:, :-1, 4] >= out[:, 1:, 4]).all()
    assert torch.equal(out[..., 5], out[..., 5].round())


def test_wrapper_classic_output_layout():
    model = tiny("detect")
    out = ExportWrapper(model)(torch.rand(1, 3, SIZE, SIZE))
    assert out.shape == (1, (SIZE // 8) ** 2 + (SIZE // 16) ** 2 + (SIZE // 32) ** 2, 4 + 3)


@pytest.mark.parametrize("head", ["dual", "detect"])
def test_onnx_runtime_parity(head: str, tmp_path: Path):
    ort = pytest.importorskip("onnxruntime")
    onnx = pytest.importorskip("onnx")
    model = tiny(head).fuse()
    path = export_onnx(model, tmp_path / f"{head}.onnx", img_size=SIZE, max_det=20)
    graph = onnx.load(str(path))
    onnx.checker.check_model(graph)
    ops = {node.op_type for node in graph.graph.node}
    assert not any(op.startswith(("Custom", "Plugin")) for op in ops)
    assert all(node.domain in ("", "ai.onnx") for node in graph.graph.node), "custom domains found"

    x = torch.rand(2, 3, SIZE, SIZE)
    with torch.no_grad():
        expected = ExportWrapper(model, 20)(x).numpy()
    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    (got,) = session.run(None, {"images": x.numpy()})
    assert got.shape == expected.shape
    assert np.abs(got - expected).max() < 1e-4, np.abs(got - expected).max()


def test_torchscript_parity(tmp_path: Path):
    model = tiny("dual").fuse()
    path = export_torchscript(model, tmp_path / "m.torchscript", img_size=SIZE, max_det=20)
    loaded = cast(torch.nn.Module, torch.jit.load(str(path)))
    x = torch.rand(1, 3, SIZE, SIZE)
    with torch.no_grad():
        expected = ExportWrapper(model, 20)(x)
        got = cast(torch.Tensor, loaded(x))
    assert torch.allclose(got, expected, atol=1e-5)


def test_model_export_method(tmp_path: Path):
    pytest.importorskip("onnx")
    model = tiny("dual")
    out = model.export("onnx", tmp_path / "x.onnx", img_size=SIZE)
    assert out.exists()
    with pytest.raises(ValueError, match="unknown export format"):
        model.export("pdf", tmp_path / "x.pdf")  # type: ignore[arg-type]  # deliberately wrong literal
