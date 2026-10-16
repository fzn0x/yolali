# Box ops and anchors

`yolali.ops` holds the pure functions every other package builds on. None
of them has parameters or state, so each is tested against numbers worked
out by hand in the test file (`tests/ops/`).

## Formats

| Function | In | Out |
|---|---|---|
| `xyxy_to_cxcywh`, `cxcywh_to_xyxy` | `(..., 4)` | `(..., 4)` |
| `ltrb_to_xyxy(anchors, ltrb)` | `(..., 2)` anchors, `(..., 4)` distances | `(..., 4)` xyxy |
| `xyxy_to_ltrb(anchors, boxes)` | inverse of the above | |

`ltrb` is the parameterisation the head predicts: distances from an anchor
point to the left, top, right and bottom sides, in stride units until
`decode_boxes` multiplies them out.

## IoU family

`broadcast_iou(a, b, kind)` is the one place the formulas live;
`box_iou` (all pairs, `(N, M)`) and `elementwise_iou` (row-wise) are thin
wrappers. `kind` is `"iou"`, `"giou"` or `"ciou"`. CIoU follows Zheng et
al. 2020 with the aspect-ratio weight `alpha` computed under `no_grad`.

A worked case from the tests: a 10×10 square at the origin and the same
square shifted by (5, 5) overlap on 25 px² of a 175 px² union, so IoU is
1/7; the enclosing box is 15×15, so GIoU is `1/7 − 50/225`; the centres
are 7.07 px apart against a 21.2 px enclosing diagonal, so CIoU is
`1/7 − 50/450`.

## Anchors

`make_anchors(feat_shapes, strides, offset=0.5)` returns `(A, 2)` cell
centres **in pixels** and `(A, 1)` strides, levels concatenated in stride
order, each level row-major. For a `(2, 2)` map at stride 8 and a `(1, 1)`
map at stride 16 the anchors are `(4,4) (12,4) (4,12) (12,12) (8,8)`.

## Decoding

`decode_boxes(box_raw, anchors, strides, reg_max)` turns raw head output
into `(B, A, 4)` pixel boxes:

- `reg_max > 0`: `dfl_expectation` takes a softmax over each side's bins
  and its expectation, in stride units (YOLOv9's `DDetect` layout, side-major).
- `reg_max == 0`: `direct_distances` applies `softplus` (the choice is
  recorded as deviation D2 in the [design](../design.md)).

Both paths multiply by the anchor's stride and call `ltrb_to_xyxy`. The
loss, the evaluator, `Detector` and the ONNX export all call this one
function, so they cannot disagree.

## NMS

`batched_nms(boxes, scores, classes, iou_threshold, max_det)` wraps
`torchvision.ops.batched_nms` and returns kept indices in descending
score. It is used only by the classic one-to-many head path; the dual head
never calls it.

Source: `src/yolali/ops/`. API: [yolali.ops](../api/ops.md).
