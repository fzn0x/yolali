"""GELAN backbone (Generalized Efficient Layer Aggregation Network).

The topology of GELAN-C from YOLOv9 (Wang, Yeh, Liao 2024, arXiv:2402.13616,
reference config ``models/detect/gelan-c.yaml``), written as named attributes
so ``print(model.backbone)`` reads as a network diagram. N/S/M variants are
width/depth scalings of this topology (``docs/design.md`` D6).
"""

from __future__ import annotations

from torch import Tensor

from yolali.nn.blocks import ADown, Conv, RepCSPELAN, make_divisible
from yolali.nn.interfaces import Backbone

# GELAN-C channel plan. Stages 2..5 are (c_out, c_mid, c_stage) of RepCSPELAN.
_STEM_CHANNELS = (64, 128)
_STAGE_CHANNELS = ((256, 128, 64), (512, 256, 128), (512, 512, 256), (512, 512, 256))


def scale_channels(c: int, width: float) -> int:
    """Width-scale a channel count, rounded up to a multiple of 8."""
    return make_divisible(c * width, 8)


def scale_depth(n: int, depth: float) -> int:
    """Depth-scale a repeat count, never below 1."""
    return max(round(n * depth), 1)


class GELANBackbone(Backbone):
    """``(B, 3, H, W)`` -> ``[P3 (B, C3, H/8, W/8), P4 (/16), P5 (/32)]``.

    Args:
        width: channel multiplier (1.0 = GELAN-C).
        depth: bottleneck-repeat multiplier (1.0 = GELAN-C, where every
            RepCSP has one bottleneck).
        in_channels: image channels.
    """

    def __init__(self, width: float = 1.0, depth: float = 1.0, in_channels: int = 3) -> None:
        super().__init__()
        c = [scale_channels(v, width) for v in _STEM_CHANNELS]
        stages = [tuple(scale_channels(v, width) for v in s) for s in _STAGE_CHANNELS]
        n = scale_depth(1, depth)
        (o2, m2, s2), (o3, m3, s3), (o4, m4, s4), (o5, m5, s5) = stages

        self.stem1 = Conv(in_channels, c[0], 3, 2)  # P1 /2
        self.stem2 = Conv(c[0], c[1], 3, 2)  # P2 /4
        self.stage2 = RepCSPELAN(c[1], o2, m2, s2, n)
        self.down3 = ADown(o2, o2)  # P3 /8
        self.stage3 = RepCSPELAN(o2, o3, m3, s3, n)
        self.down4 = ADown(o3, o3)  # P4 /16
        self.stage4 = RepCSPELAN(o3, o4, m4, s4, n)
        self.down5 = ADown(o4, o4)  # P5 /32
        self.stage5 = RepCSPELAN(o4, o5, m5, s5, n)

        self.out_channels: tuple[int, int, int] = (o3, o4, o5)

    def forward(self, x: Tensor) -> list[Tensor]:
        """Run the stem and four stages, returning the last three stage outputs."""
        x = self.stage2(self.stem2(self.stem1(x)))
        p3 = self.stage3(self.down3(x))
        p4 = self.stage4(self.down4(p3))
        p5 = self.stage5(self.down5(p4))
        return [p3, p4, p5]
