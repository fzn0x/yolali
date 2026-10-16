# Assigners

An assigner decides which anchors are responsible for which ground-truth
box. Both assigners take post-sigmoid class scores `(B, A, nc)`, decoded
boxes `(B, A, 4)`, anchors `(A, 2)` and `Targets`, and return an
`Assignment`:

| Field | Shape | Meaning |
|---|---|---|
| `boxes` | `(B, A, 4)` | the box each positive anchor must regress to |
| `scores` | `(B, A, nc)` | soft classification targets in `[0, 1]` |
| `fg_mask` | `(B, A)` | which anchors are positive |
| `gt_idx` | `(B, A)` | index of the matched ground truth |

## TaskAlignedAssigner (one-to-many)

The alignment metric of TOOD, as used by the YOLOv8/v9 lineage:
`score[gt_class]^alpha · iou^beta` with `alpha = 0.5`, `beta = 6`. The
forward pass reads top to bottom in `src/yolali/assign/tal.py`:

1. `in_gt`: which anchors lie inside which box.
2. IoU between every ground truth and every predicted box.
3. The predicted probability of each ground truth's class at every anchor.
4. The metric, zero outside the box.
5. Top-`k` (default 10) anchors per ground truth become candidates.
6. An anchor claimed by several boxes goes to the one with the highest
   **IoU** (deviation D1).
7. Gather the target box and class per positive anchor.
8. Soft targets: the metric is rescaled per box so its best anchor gets
   that box's best IoU.

The toy case in `tests/assign/test_tal.py` is a 4×4 grid at stride 8 where
every predicted box is an 8×8 square centred on its anchor; an anchor
inside a 16×16 box therefore has IoU exactly 0.25, and the expected soft
targets are `sqrt(score) · 0.25`.

## OneToOneAssigner

YOLOv10's consistent dual assignment: the same metric with `topk = 1`, so
each ground truth receives at most one anchor. A box can receive none when
its best anchor is also the best anchor of a box with higher IoU; that is
accepted, as in YOLOv10. Because both heads agree on what "best" means,
the one-to-one head learns to put its confidence where the one-to-many
head's top candidate sits, and inference can skip NMS.

Source: `src/yolali/assign/`. API: [yolali.assign](../api/assign.md).
