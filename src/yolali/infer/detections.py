"""The result type of :meth:`yolali.infer.Detector.predict`."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass(frozen=True)
class Detections:
    """Boxes for one image, in pixels of the **original** image.

    Attributes:
        xyxy: ``(N, 4)`` float32.
        conf: ``(N,)`` float32 in ``[0, 1]``.
        cls: ``(N,)`` int64 class indices.
        names: optional class names, indexed by ``cls``.
        orig_hw: ``(H, W)`` of the image the boxes refer to.
    """

    xyxy: Tensor
    conf: Tensor
    cls: Tensor
    names: tuple[str, ...] | None = None
    orig_hw: tuple[int, int] = (0, 0)

    def __post_init__(self) -> None:
        if self.xyxy.ndim != 2 or self.xyxy.shape[1] != 4:
            raise ValueError(f"xyxy must be (N, 4), got {tuple(self.xyxy.shape)}")
        if self.conf.shape != self.xyxy.shape[:1] or self.cls.shape != self.xyxy.shape[:1]:
            raise ValueError("conf and cls must be (N,) matching xyxy")

    def __len__(self) -> int:
        return int(self.xyxy.shape[0])

    def __iter__(self) -> Iterator[tuple[Tensor, float, int]]:
        """Yield ``(xyxy, conf, cls)`` per box."""
        for i in range(len(self)):
            yield self.xyxy[i], float(self.conf[i]), int(self.cls[i])

    def label(self, i: int) -> str:
        """Class name (or index) of box ``i``."""
        c = int(self.cls[i])
        return self.names[c] if self.names is not None and c < len(self.names) else str(c)

    def filter(self, min_conf: float) -> Detections:
        """Keep boxes with ``conf >= min_conf``."""
        keep = self.conf >= min_conf
        return Detections(
            self.xyxy[keep], self.conf[keep], self.cls[keep], self.names, self.orig_hw
        )

    def to(self, device: torch.device | str) -> Detections:
        """Move tensors."""
        return Detections(
            self.xyxy.to(device),
            self.conf.to(device),
            self.cls.to(device),
            self.names,
            self.orig_hw,
        )

    @classmethod
    def empty(
        cls, names: tuple[str, ...] | None = None, orig_hw: tuple[int, int] = (0, 0)
    ) -> Detections:
        """No boxes."""
        return cls(
            torch.zeros(0, 4), torch.zeros(0), torch.zeros(0, dtype=torch.int64), names, orig_hw
        )
