"""Model configuration: flat YAML -> frozen dataclasses -> modules.

YAML describes *shape* only::

    name: gelan - n
    nc: 80
    backbone: {type: gelan, width: 0.25, depth: 0.33}
    neck: {type: gelan_pan, width: 0.25, depth: 0.33}
    head: {type: detect, reg_max: 16}

Builders are looked up in a plain dictionary passed to :func:`build_model`,
so adding a backbone means adding a key, not registering a decorator at
import time.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml
from torch import nn

from yolali.nn.backbone.gelan import GELANBackbone
from yolali.nn.head.detect import DetectHead
from yolali.nn.head.dual import DualDetectHead
from yolali.nn.interfaces import Backbone, Channels, Neck, Strides
from yolali.nn.neck.gelan_pan import GELANPAN


@dataclass(frozen=True)
class ComponentConfig:
    """One ``{type: ..., **options}`` mapping from the YAML."""

    type: str
    options: Mapping[str, Any] = field(default_factory=dict[str, Any])

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any], section: str) -> ComponentConfig:
        """Split ``type`` from the remaining options; raise a readable error if missing."""
        if "type" not in raw:
            raise ValueError(f"'{section}' section needs a 'type' key")
        options: dict[str, Any] = {str(k): v for k, v in raw.items() if k != "type"}
        return cls(str(raw["type"]), options)

    def get_float(self, key: str, default: float) -> float:
        """Read a float option with a default."""
        return float(self.options.get(key, default))

    def get_int(self, key: str, default: int) -> int:
        """Read an int option with a default."""
        return int(self.options.get(key, default))


@dataclass(frozen=True)
class ModelConfig:
    """Shape of a detector."""

    name: str
    num_classes: int
    backbone: ComponentConfig
    neck: ComponentConfig
    head: ComponentConfig

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> ModelConfig:
        """Validate a parsed YAML mapping."""
        for key in ("name", "nc", "backbone", "neck", "head"):
            if key not in raw:
                raise ValueError(f"model config is missing '{key}'")
        return cls(
            name=str(raw["name"]),
            num_classes=int(raw["nc"]),
            backbone=ComponentConfig.from_mapping(raw["backbone"], "backbone"),
            neck=ComponentConfig.from_mapping(raw["neck"], "neck"),
            head=ComponentConfig.from_mapping(raw["head"], "head"),
        )

    @classmethod
    def from_yaml(cls, path: str | Path) -> ModelConfig:
        """Load and validate a YAML file."""
        with Path(path).open(encoding="utf-8") as f:
            raw = yaml.safe_load(f)
        if not isinstance(raw, Mapping):
            raise ValueError(f"{path} must contain a mapping at the top level")
        return cls.from_mapping(raw)  # pyright: ignore[reportUnknownArgumentType] - yaml returns Any


BackboneBuilder = Callable[[ComponentConfig], Backbone]
NeckBuilder = Callable[[ComponentConfig, Channels], Neck]
HeadBuilder = Callable[[ComponentConfig, Channels, int, Strides], nn.Module]


def _build_gelan(cfg: ComponentConfig) -> Backbone:
    return GELANBackbone(cfg.get_float("width", 1.0), cfg.get_float("depth", 1.0))


def _build_gelan_pan(cfg: ComponentConfig, in_channels: Channels) -> Neck:
    return GELANPAN(in_channels, cfg.get_float("width", 1.0), cfg.get_float("depth", 1.0))


def _build_detect(
    cfg: ComponentConfig, in_channels: Channels, nc: int, strides: Strides
) -> nn.Module:
    return DetectHead(in_channels, nc, cfg.get_int("reg_max", 16), strides)


def _build_vit(cfg: ComponentConfig) -> Backbone:
    from yolali.nn.backbone.vit import ViTBackbone  # noqa: PLC0415 - keeps timm optional

    channels = cfg.options.get("out_channels", (128, 256, 384))
    c3, c4, c5 = (int(c) for c in channels)
    return ViTBackbone(
        str(cfg.options.get("model", "vit_small_patch14_dinov2.lvd142m")),
        (c3, c4, c5),
        img_size=cfg.get_int("img_size", 640),
        patch_size=cfg.get_int("patch_size", 16),
        pretrained=bool(cfg.options.get("pretrained", True)),
        freeze=bool(cfg.options.get("freeze", False)),
    )


BACKBONES: dict[str, BackboneBuilder] = {"gelan": _build_gelan, "vit": _build_vit}
NECKS: dict[str, NeckBuilder] = {"gelan_pan": _build_gelan_pan}


def _build_dual(
    cfg: ComponentConfig, in_channels: Channels, nc: int, strides: Strides
) -> nn.Module:
    return DualDetectHead(
        in_channels,
        nc,
        cfg.get_int("reg_max", 0),
        strides,
        detach_o2o=bool(cfg.options.get("detach_o2o", True)),
    )


HEADS: dict[str, HeadBuilder] = {"detect": _build_detect, "dual": _build_dual}
