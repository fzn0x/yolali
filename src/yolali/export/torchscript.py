"""TorchScript export via tracing of the same :class:`ExportWrapper` used for ONNX."""

from __future__ import annotations

from pathlib import Path
from typing import cast

import torch

from yolali.export.onnx import ExportWrapper
from yolali.nn.model import YolaliModel


def export_torchscript(
    model: YolaliModel,
    path: str | Path,
    *,
    img_size: int | tuple[int, int] = 640,
    max_det: int = 300,
) -> Path:
    """Trace the wrapped model at a fixed resolution and save it."""
    path = Path(path)
    h, w = (img_size, img_size) if isinstance(img_size, int) else img_size
    wrapper = ExportWrapper(model, max_det).cpu().eval()
    with torch.no_grad():
        traced = cast(torch.jit.ScriptModule, torch.jit.trace(wrapper, torch.zeros(1, 3, h, w)))
    traced.save(str(path))
    return path
