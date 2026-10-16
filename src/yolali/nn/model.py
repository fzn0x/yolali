"""The detector: backbone -> neck -> head, and nothing else."""

from __future__ import annotations

from pathlib import Path

import torch
from torch import Tensor, nn

from yolali.nn.blocks import fuse_all
from yolali.nn.config import (
    BACKBONES,
    HEADS,
    NECKS,
    BackboneBuilder,
    HeadBuilder,
    ModelConfig,
    NeckBuilder,
)
from yolali.ops.decode import decode_boxes
from yolali.types import DualRawPrediction, RawPrediction


class YolaliModel(nn.Module):
    """``(B, 3, H, W)`` -> :class:`RawPrediction`.

    Three attributes, one ``forward``. Build it from a config with
    :func:`build_model` or assemble the parts yourself.
    """

    def __init__(
        self, backbone: nn.Module, neck: nn.Module, head: nn.Module, cfg: ModelConfig
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.neck = neck
        self.head = head
        self.cfg = cfg

    @property
    def num_classes(self) -> int:
        """``nc`` from the config."""
        return self.cfg.num_classes

    @property
    def reg_max(self) -> int:
        """DFL bins of the head (``0`` for direct regression)."""
        return self.cfg.head.get_int("reg_max", 0 if self.cfg.head.type == "dual" else 16)

    @property
    def nms_free(self) -> bool:
        """``True`` when inference uses the one-to-one head and needs no NMS."""
        return self.cfg.head.type == "dual"

    def forward(self, x: Tensor) -> RawPrediction | DualRawPrediction:
        """Raw per-anchor predictions; decode with :meth:`decode`.

        A dual head returns :class:`DualRawPrediction` while training and a
        plain :class:`RawPrediction` (its one-to-one branch) in ``eval()``.
        """
        return self.head(self.neck(self.backbone(x)))

    def decode(self, pred: RawPrediction) -> tuple[Tensor, Tensor]:
        """``RawPrediction`` -> ``(boxes (B, A, 4) xyxy px, scores (B, A, nc) in [0, 1])``."""
        boxes = decode_boxes(pred.box_raw, pred.anchors, pred.strides, self.reg_max)
        return boxes, pred.cls_logits.sigmoid()

    def fuse(self) -> YolaliModel:
        """Fold BatchNorm and RepConv branches for inference. Returns ``self``."""
        for child in self.children():
            fuse_all(child)
        return self


def build_model(
    cfg: ModelConfig,
    *,
    backbones: dict[str, BackboneBuilder] = BACKBONES,
    necks: dict[str, NeckBuilder] = NECKS,
    heads: dict[str, HeadBuilder] = HEADS,
) -> YolaliModel:
    """Instantiate a :class:`YolaliModel` from a config and builder tables."""
    try:
        backbone = backbones[cfg.backbone.type](cfg.backbone)
        neck = necks[cfg.neck.type](cfg.neck, backbone.out_channels)
        head = heads[cfg.head.type](cfg.head, neck.out_channels, cfg.num_classes, backbone.strides)
    except KeyError as exc:
        raise ValueError(f"unknown component type {exc}") from exc
    return YolaliModel(backbone, neck, head, cfg)


def load_model(
    config: str | Path, weights: str | Path | None = None, device: str | torch.device = "cpu"
) -> YolaliModel:
    """Build from a YAML config and optionally load a yolali ``state_dict`` file."""
    model = build_model(ModelConfig.from_yaml(config))
    if weights is not None:
        state = torch.load(weights, map_location="cpu", weights_only=True)
        if not isinstance(state, dict):
            raise ValueError(f"{weights} does not contain a state_dict")
        model.load_state_dict(state)  # pyright: ignore[reportUnknownArgumentType] - torch.load returns Any
    return model.to(device)
