"""Spatial Tuning Adapter: one ``1/16`` ViT feature map -> ``[P3, P4, P5]``.

After DEIMv2 (Huang et al. 2025, arXiv:2509.20787), which showed that a
plain-ViT backbone (DINOv3) needs only a light adapter to feed a multi-scale
detector neck. We implement the simplest pyramid that satisfies the neck
interface (``docs/design.md`` D4):

* ``P3``: ``ConvTranspose2d(k=2, s=2)`` doubles the resolution, then ``Conv3x3``.
* ``P4``: ``Conv1x1`` projection at native resolution, then ``Conv3x3``.
* ``P5``: ``Conv3x3`` with stride 2, then ``Conv3x3``.

No cross-scale fusion happens here; that is the neck's job.
"""

from __future__ import annotations

from torch import Tensor, nn

from yolali.nn.blocks import Conv
from yolali.nn.interfaces import Channels


class SpatialTuningAdapter(nn.Module):
    """One ``(B, C_in, H/16, W/16)`` map -> ``[P3 /8, P4 /16, P5 /32]`` with ``out_channels``.

    Args:
        c_in: transformer embedding dimension.
        out_channels: ``(c3, c4, c5)``.
    """

    def __init__(self, c_in: int, out_channels: Channels) -> None:
        super().__init__()
        c3, c4, c5 = out_channels
        self.up = nn.ConvTranspose2d(c_in, c3, kernel_size=2, stride=2)
        self.p3 = Conv(c3, c3, 3)
        self.proj = Conv(c_in, c4, 1)
        self.p4 = Conv(c4, c4, 3)
        self.down = Conv(c_in, c5, 3, 2)
        self.p5 = Conv(c5, c5, 3)
        self.out_channels: Channels = out_channels

    def forward(self, x: Tensor) -> list[Tensor]:
        """Build the three scales from one map."""
        return [self.p3(self.up(x)), self.p4(self.proj(x)), self.p5(self.down(x))]
