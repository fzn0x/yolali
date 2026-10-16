"""ONNX export with standard ops only.

The exported graph includes decoding. For an NMS-free model the output is
already the final detections ``(B, max_det, 6)`` = ``[x1, y1, x2, y2, conf,
cls]`` in letterboxed pixels; the only "post-processing" is a ``TopK`` over
per-anchor max scores, which is a core ONNX op. For a classic head the
output is ``(B, A, 4 + nc)`` (boxes then class probabilities) and the caller
runs NMS.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import torch
from torch import Tensor, nn

from yolali.nn.model import YolaliModel
from yolali.types import RawPrediction


class ExportWrapper(nn.Module):
    """``(B, 3, H, W)`` -> detections or raw boxes+scores, see module docstring."""

    def __init__(self, model: YolaliModel, max_det: int = 300) -> None:
        super().__init__()
        self.model = model.eval()
        self.max_det = max_det
        self.nms_free = model.nms_free

    def forward(self, x: Tensor) -> Tensor:
        """Run the detector and decode."""
        pred = self.model(x)
        assert isinstance(pred, RawPrediction)  # eval-mode heads always return one branch
        boxes, scores = self.model.decode(pred)  # (B, A, 4), (B, A, nc)
        if not self.nms_free:
            return torch.cat((boxes, scores), 2)
        conf, cls = scores.max(2)  # (B, A)
        k = min(self.max_det, int(conf.shape[1]))
        top_conf, idx = conf.topk(k, dim=1)  # (B, k)
        top_boxes = boxes.gather(1, idx.unsqueeze(-1).expand(-1, -1, 4))
        top_cls = cls.gather(1, idx).to(boxes.dtype)
        return torch.cat((top_boxes, top_conf.unsqueeze(-1), top_cls.unsqueeze(-1)), 2)


def export_onnx(
    model: YolaliModel,
    path: str | Path,
    *,
    img_size: int | tuple[int, int] = 640,
    max_det: int = 300,
    opset: int = 18,
    dynamic_batch: bool = True,
    simplify: bool = False,
) -> Path:
    """Write ``path`` and return it.

    Args:
        model: weights loaded; fused or not (fusing first makes a smaller graph).
        path: output ``.onnx``.
        img_size: ``H`` or ``(H, W)`` of the fixed input resolution.
        max_det: detections kept by the NMS-free output.
        opset: ONNX opset.
        dynamic_batch: mark the batch dimension dynamic.
        simplify: run ``onnxsim`` when installed (optional).
    """
    path = Path(path)
    h, w = (img_size, img_size) if isinstance(img_size, int) else img_size
    wrapper = ExportWrapper(model, max_det).cpu().eval()
    dummy = torch.zeros(1, 3, h, w)
    dynamic_axes = {"images": {0: "batch"}, "output": {0: "batch"}} if dynamic_batch else None
    torch.onnx.export(
        wrapper,
        (dummy,),
        str(path),
        input_names=["images"],
        output_names=["output"],
        opset_version=opset,
        dynamic_axes=dynamic_axes,
        dynamo=True,
        external_data=False,
    )
    if simplify:
        try:
            import onnx  # noqa: PLC0415 - optional dependency
            import onnxsim  # noqa: PLC0415 # pyright: ignore[reportMissingImports] - optional dependency
        except ImportError as exc:
            raise ImportError("simplify=True needs the `onnxsim` package") from exc
        simplified, ok = cast(tuple[Any, bool], onnxsim.simplify(onnx.load(str(path))))
        if ok:
            onnx.save(simplified, str(path))
    return path
