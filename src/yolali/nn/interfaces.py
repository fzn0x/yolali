"""Base classes that fix the contract between backbone, neck and head.

Both are ordinary ``nn.Module`` subclasses; they exist so a reader (and the
type checker) can see exactly what a pluggable component must provide.
"""

from __future__ import annotations

from torch import Tensor, nn

Channels = tuple[int, int, int]
Strides = tuple[int, int, int]


class Backbone(nn.Module):
    """Image -> three feature maps at strides ``(8, 16, 32)``.

    Subclasses set ``out_channels`` in ``__init__`` and may override ``strides``.
    """

    out_channels: Channels
    strides: Strides = (8, 16, 32)

    def forward(self, x: Tensor) -> list[Tensor]:
        """``(B, 3, H, W)`` -> ``[P3, P4, P5]``."""
        raise NotImplementedError


class Neck(nn.Module):
    """``[P3, P4, P5]`` -> ``[N3, N4, N5]`` at the same strides.

    Subclasses set ``out_channels`` in ``__init__``.
    """

    out_channels: Channels

    def forward(self, feats: list[Tensor]) -> list[Tensor]:
        """Fuse multi-scale features."""
        raise NotImplementedError
