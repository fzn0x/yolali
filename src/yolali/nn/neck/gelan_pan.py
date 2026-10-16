"""GELAN path-aggregation neck (the "head" section of the YOLOv9 GELAN-C config).

Top-down: SPP on P5, upsample and concatenate with P4 then P3, each merge a
:class:`RepCSPELAN`. Bottom-up: :class:`ADown` and concatenate with the
top-down result at the next scale. Reference: Wang, Yeh, Liao 2024,
arXiv:2402.13616, ``models/detect/gelan-c.yaml`` layers 9-21.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F  # noqa: N812 - torch convention
from torch import Tensor

from yolali.nn.backbone.gelan import scale_channels, scale_depth
from yolali.nn.blocks import SPPELAN, ADown, RepCSPELAN
from yolali.nn.interfaces import Neck

# GELAN-C neck channel plan: (c_out, c_mid, c_stage) per merge block, plus SPP width.
_SPP = (512, 256)  # (c_out, c_mid)
_UP_P4 = (512, 512, 256)
_UP_P3 = (256, 256, 128)
_OUT_P4 = (512, 512, 256)
_OUT_P5 = (512, 512, 256)


class GELANPAN(Neck):
    """``[P3, P4, P5]`` -> ``[N3, N4, N5]`` with channels ``out_channels``.

    Args:
        in_channels: channels of ``(P3, P4, P5)`` from the backbone.
        width: channel multiplier applied to the GELAN-C neck plan.
        depth: bottleneck-repeat multiplier.
    """

    def __init__(
        self, in_channels: tuple[int, int, int], width: float = 1.0, depth: float = 1.0
    ) -> None:
        super().__init__()
        c3_in, c4_in, c5_in = in_channels
        n = scale_depth(1, depth)
        spp_out, spp_mid = (scale_channels(v, width) for v in _SPP)
        up4 = tuple(scale_channels(v, width) for v in _UP_P4)
        up3 = tuple(scale_channels(v, width) for v in _UP_P3)
        out4 = tuple(scale_channels(v, width) for v in _OUT_P4)
        out5 = tuple(scale_channels(v, width) for v in _OUT_P5)

        self.spp = SPPELAN(c5_in, spp_out, spp_mid)
        self.up_p4 = RepCSPELAN(spp_out + c4_in, up4[0], up4[1], up4[2], n)
        self.up_p3 = RepCSPELAN(up4[0] + c3_in, up3[0], up3[1], up3[2], n)
        self.down_p4 = ADown(up3[0], up3[0])
        self.out_p4 = RepCSPELAN(up3[0] + up4[0], out4[0], out4[1], out4[2], n)
        self.down_p5 = ADown(out4[0], out4[0])
        self.out_p5 = RepCSPELAN(out4[0] + spp_out, out5[0], out5[1], out5[2], n)

        self.out_channels: tuple[int, int, int] = (up3[0], out4[0], out5[0])

    def forward(self, feats: list[Tensor]) -> list[Tensor]:
        """Fuse three scales; see module docstring for the data flow."""
        p3, p4, p5 = feats
        t5 = self.spp(p5)  # (B, spp_out, H/32, W/32)
        t4 = self.up_p4(torch.cat((F.interpolate(t5, scale_factor=2.0, mode="nearest"), p4), 1))
        n3 = self.up_p3(torch.cat((F.interpolate(t4, scale_factor=2.0, mode="nearest"), p3), 1))
        n4 = self.out_p4(torch.cat((self.down_p4(n3), t4), 1))
        n5 = self.out_p5(torch.cat((self.down_p5(n4), t5), 1))
        return [n3, n4, n5]
