"""Run a model over a dataloader and score it."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import cast

import torch
from torch import Tensor, nn

from yolali.data.collate import Batch
from yolali.data.transforms.letterbox import undo_letterbox
from yolali.eval.map import DetectionMetrics, MetricAccumulator
from yolali.ops.nms import batched_nms

Detections = tuple[Tensor, Tensor, Tensor]  # boxes (N, 4), scores (N,), classes (N,)
GroundTruth = tuple[Tensor, Tensor, Tensor, Tensor]  # boxes, classes, crowd, areas


@dataclass(frozen=True)
class EvalConfig:
    """How predictions become scored detections.

    Attributes:
        conf: minimum score; the COCO protocol uses a very low value so the
            PR curve reaches high recall.
        iou: NMS IoU threshold (classic head only).
        max_det: detections kept per image before scoring.
        nms: run class-aware NMS (``True`` for the one-to-many head, ``False``
            for the NMS-free one-to-one head).
        half: run the model in float16 on CUDA.
    """

    conf: float = 0.001
    iou: float = 0.7
    max_det: int = 300
    nms: bool = True
    half: bool = False


def postprocess(
    boxes: Tensor, scores: Tensor, *, conf: float, iou: float, max_det: int, nms: bool
) -> list[Detections]:
    """``(B, A, 4)`` boxes and ``(B, A, nc)`` scores -> per-image ``(boxes, scores, classes)``.

    Every (anchor, class) pair above ``conf`` is a candidate, as in the COCO
    multi-label protocol. With ``nms=True`` candidates are suppressed per
    class; otherwise the top ``max_det`` by score are kept directly.
    """
    out: list[Detections] = []
    for b in range(boxes.shape[0]):
        anchor_idx, cls_idx = (scores[b] > conf).nonzero(as_tuple=True)
        s = scores[b][anchor_idx, cls_idx]
        bx = boxes[b][anchor_idx]
        if nms:
            keep = batched_nms(bx, s, cls_idx, iou, max_det)
        else:
            keep = torch.argsort(-s, stable=True)[:max_det]
        out.append((bx[keep], s[keep], cls_idx[keep]))
    return out


class Evaluator:
    """Score a detector on batches that carry letterbox metadata.

    Args:
        model: a module with ``forward(images)`` and ``decode(pred) -> (boxes, scores)``.
        num_classes: ``nc``.
        cfg: postprocessing settings.
        device: where to run.
        gt_lookup: maps an image path to ``(boxes, classes, crowd, areas)`` in
            original pixels. Defaults to the batch targets mapped back through
            the letterbox, which has no crowd flags or annotation areas; for
            COCO numbers pass a lookup built from the json.
    """

    def __init__(
        self,
        model: nn.Module,
        num_classes: int,
        cfg: EvalConfig | None = None,
        device: torch.device | str = "cpu",
        gt_lookup: Callable[[str], GroundTruth] | None = None,
    ) -> None:
        self.model = model
        self.num_classes = num_classes
        self.cfg = cfg or EvalConfig()
        self.device = torch.device(device)
        self.gt_lookup = gt_lookup

    @torch.no_grad()
    def predict_batch(self, batch: Batch) -> list[Detections]:
        """Forward + postprocess + undo letterbox -> detections in original pixels."""
        images = batch.images.to(self.device, non_blocking=True)
        use_half = self.cfg.half and self.device.type == "cuda"
        with torch.autocast(self.device.type, dtype=torch.float16, enabled=use_half):
            pred = self.model(images)
        decode = getattr(self.model, "decode", None)
        if not callable(decode):
            raise TypeError("model must provide decode(RawPrediction) -> (boxes, scores)")
        boxes, scores = cast(tuple[Tensor, Tensor], decode(pred))
        dets = postprocess(
            boxes.float(),
            scores.float(),
            conf=self.cfg.conf,
            iou=self.cfg.iou,
            max_det=self.cfg.max_det,
            nms=self.cfg.nms,
        )
        out: list[Detections] = []
        for i, (bx, s, c) in enumerate(dets):
            restored = undo_letterbox(bx.cpu(), batch.scales[i], batch.pads[i], batch.orig_hw[i])
            out.append((restored, s.cpu(), c.cpu()))
        return out

    def ground_truth(self, batch: Batch, i: int) -> GroundTruth:
        """Ground truth of image ``i`` in original pixels."""
        if self.gt_lookup is not None:
            return self.gt_lookup(batch.paths[i])
        mask = batch.targets.mask[i]
        boxes = undo_letterbox(
            batch.targets.boxes[i][mask].cpu(), batch.scales[i], batch.pads[i], batch.orig_hw[i]
        )
        classes = batch.targets.classes[i][mask].cpu()
        crowd = torch.zeros(boxes.shape[0], dtype=torch.bool)
        areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        return boxes, classes, crowd, areas

    def run(
        self, loader: Iterable[Batch], progress: Callable[[int], None] | None = None
    ) -> DetectionMetrics:
        """Evaluate every batch and return metrics."""
        was_training = self.model.training
        self.model.eval()
        acc = MetricAccumulator(self.num_classes, max_det=100)
        for n, batch in enumerate(loader):
            for i, (bx, s, c) in enumerate(self.predict_batch(batch)):
                gt_boxes, gt_classes, gt_crowd, gt_areas = self.ground_truth(batch, i)
                acc.add_image(bx, s, c, gt_boxes, gt_classes, gt_crowd=gt_crowd, gt_areas=gt_areas)
            if progress is not None:
                progress(n + 1)
        if was_training:
            self.model.train()
        return acc.compute()
