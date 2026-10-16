"""Anything image-like -> one letterboxed batch tensor plus the metadata to undo it."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch import Tensor

from yolali.data.dataset import load_image
from yolali.data.sample import Sample
from yolali.data.transforms.letterbox import Letterbox

ImageInput = str | Path | np.ndarray | Image.Image | Tensor


@dataclass(frozen=True)
class LetterboxMeta:
    """What :func:`undo_letterbox` needs for one image."""

    orig_hw: tuple[int, int]
    scale: float
    pad: tuple[float, float]


def to_rgb_array(image: ImageInput) -> np.ndarray:
    """Path, PIL image, ``(H, W, 3)`` uint8 array or ``(3, H, W)`` tensor -> uint8 RGB array."""
    if isinstance(image, (str, Path)):
        return load_image(image)
    if isinstance(image, Image.Image):
        return np.asarray(image.convert("RGB"), dtype=np.uint8)
    if isinstance(image, Tensor):
        if image.ndim != 3 or image.shape[0] != 3:
            raise ValueError(f"tensor images must be (3, H, W), got {tuple(image.shape)}")
        arr = image.detach().cpu().float()
        if arr.max() <= 1.0:
            arr = arr * 255.0
        return arr.clamp(0, 255).permute(1, 2, 0).to(torch.uint8).numpy()
    arr = np.asarray(image)
    if arr.ndim == 2:
        arr = np.repeat(arr[..., None], 3, axis=2)
    if arr.ndim != 3 or arr.shape[2] != 3:
        raise ValueError(f"array images must be (H, W, 3), got {arr.shape}")
    return np.ascontiguousarray(arr, dtype=np.uint8)


def prepare_images(images: Sequence[ImageInput], size: int) -> tuple[Tensor, list[LetterboxMeta]]:
    """Letterbox every input to ``size x size``; ``(B, 3, size, size)`` floats in ``[0, 1]``."""
    letterbox = Letterbox(size)
    rng = np.random.default_rng(0)  # Letterbox is deterministic; the generator is unused
    tensors: list[Tensor] = []
    metas: list[LetterboxMeta] = []
    for image in images:
        arr = to_rgb_array(image)
        sample = Sample(arr, np.zeros((0, 4), np.float32), np.zeros(0, np.int64))
        boxed = letterbox(sample, rng)
        tensors.append(torch.from_numpy(boxed.image).permute(2, 0, 1))
        metas.append(LetterboxMeta(sample.orig_hw, boxed.scale, boxed.pad))
    batch = torch.stack(tensors).float().div_(255.0)
    return batch, metas
