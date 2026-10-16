"""``Detector(model).predict(images)``."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import Tensor

from yolali.data.transforms.letterbox import undo_letterbox
from yolali.eval.evaluator import postprocess
from yolali.infer.detections import Detections
from yolali.infer.preprocess import ImageInput, prepare_images
from yolali.nn.model import YolaliModel
from yolali.train.trainer import resolve_device


class Detector:
    """Batched inference on CPU, CUDA or MPS.

    Args:
        model: a :class:`YolaliModel` (weights already loaded).
        device: ``"auto"`` or an explicit device.
        half: float16 autocast on CUDA.
        fuse: fold BatchNorm / RepConv branches before running.
        names: class names attached to every :class:`Detections`.
    """

    def __init__(
        self,
        model: YolaliModel,
        device: str | torch.device = "auto",
        *,
        half: bool = False,
        fuse: bool = True,
        names: Sequence[str] | None = None,
    ) -> None:
        self.device = resolve_device(device)
        self.model = (model.fuse() if fuse else model).to(self.device).eval()
        self.half = half and self.device.type == "cuda"
        self.names = tuple(names) if names is not None else None

    @torch.no_grad()
    def predict(
        self,
        images: ImageInput | Sequence[ImageInput],
        *,
        conf: float = 0.25,
        iou: float = 0.7,
        max_det: int = 300,
        size: int = 640,
    ) -> list[Detections]:
        """Detect objects in one or many images.

        Args:
            images: a path, PIL image, ``(H, W, 3)`` uint8 RGB array or
                ``(3, H, W)`` tensor, or a sequence of those.
            conf: minimum confidence.
            iou: NMS threshold; ignored by NMS-free models.
            max_det: detections per image.
            size: letterbox size (multiple of 32).

        Returns:
            One :class:`Detections` per input, boxes in original pixels.
        """
        items: Sequence[ImageInput] = (
            [images] if isinstance(images, str) or not isinstance(images, Sequence) else images
        )
        batch, metas = prepare_images(items, size)
        batch = batch.to(self.device)
        with torch.autocast(self.device.type, dtype=torch.float16, enabled=self.half):
            pred = self.model(batch)
        boxes, scores = self.model.decode(pred)
        dets = postprocess(
            boxes.float(),
            scores.float(),
            conf=conf,
            iou=iou,
            max_det=max_det,
            nms=not self.model.nms_free,
        )
        out: list[Detections] = []
        for (bx, s, c), meta in zip(dets, metas, strict=True):
            restored: Tensor = undo_letterbox(bx.cpu(), meta.scale, meta.pad, meta.orig_hw)
            out.append(Detections(restored, s.cpu(), c.cpu(), self.names, meta.orig_hw))
        return out
