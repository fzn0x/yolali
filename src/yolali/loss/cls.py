"""Classification losses on soft per-anchor targets.

Both functions return the *sum* over all elements; the caller divides by the
total target mass so the magnitude is independent of image size.
"""

from __future__ import annotations

import torch.nn.functional as F  # noqa: N812 - torch convention
from torch import Tensor


def bce_loss(logits: Tensor, targets: Tensor) -> Tensor:
    """Binary cross-entropy with soft targets, summed.

    Args:
        logits: ``(..., nc)`` pre-sigmoid.
        targets: ``(..., nc)`` in ``[0, 1]``.
    """
    return F.binary_cross_entropy_with_logits(logits, targets, reduction="sum")


def varifocal_loss(
    logits: Tensor, targets: Tensor, alpha: float = 0.75, gamma: float = 2.0
) -> Tensor:
    """Varifocal loss (Zhang et al., CVPR 2021, arXiv:2008.13367), summed.

    Positives (``targets > 0``) are weighted by their soft target; negatives
    are down-weighted by ``alpha * p ** gamma`` so easy negatives fade out.

    Args:
        logits: ``(..., nc)`` pre-sigmoid.
        targets: ``(..., nc)`` soft targets, ``0`` for negatives.
        alpha: negative weight scale.
        gamma: focusing exponent.
    """
    prob = logits.sigmoid()
    positive = (targets > 0).to(logits.dtype)
    weight = alpha * prob.pow(gamma) * (1 - positive) + targets * positive
    per_element = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    return (per_element * weight).sum()
