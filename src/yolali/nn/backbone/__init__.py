"""Backbones: image -> ``[P3, P4, P5]`` features at strides 8, 16, 32.

``ViTBackbone`` lives in :mod:`yolali.nn.backbone.vit` and needs the ``vit``
extra; import it from there so the core package stays light.
"""

from yolali.nn.backbone.gelan import GELANBackbone
from yolali.nn.backbone.sta import SpatialTuningAdapter

__all__ = ["GELANBackbone", "SpatialTuningAdapter"]
