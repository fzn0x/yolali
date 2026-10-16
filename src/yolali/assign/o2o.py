"""One-to-one label assignment for NMS-free training.

YOLOv10 (Wang et al. 2024, arXiv:2405.14458) trains a second head with the
*same* alignment metric as the one-to-many head but keeps only the single
best anchor per ground truth. Because both heads agree on what "best" means,
the one-to-one head learns to put its confidence where the one-to-many head
would have put its top candidate, and inference can skip NMS.
"""

from __future__ import annotations

from torch import Tensor

from yolali.assign.tal import Assignment, TaskAlignedAssigner
from yolali.types import Targets


class OneToOneAssigner(TaskAlignedAssigner):
    """Task-aligned assignment with ``topk = 1``.

    Each ground-truth box receives at most one anchor. A box can receive
    none when its best anchor is also the best anchor of a box with higher
    IoU; that box is then unmatched for this step, as in YOLOv10.
    """

    def __init__(self, alpha: float = 0.5, beta: float = 6.0, eps: float = 1e-9) -> None:
        super().__init__(topk=1, alpha=alpha, beta=beta, eps=eps)

    def forward(
        self, pred_scores: Tensor, pred_boxes: Tensor, anchors: Tensor, targets: Targets
    ) -> Assignment:
        """See :meth:`TaskAlignedAssigner.forward`; guarantees one anchor per box."""
        return super().forward(pred_scores, pred_boxes, anchors, targets)
