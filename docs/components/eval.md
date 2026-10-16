# Evaluator

yolali computes COCO-protocol mAP itself, in plain PyTorch and NumPy, and
proves it against pycocotools rather than wrapping it.

## Matching

`match_predictions` reimplements the protocol per image and class:
predictions in descending score; each takes the free *regular* ground truth
of its class with the highest IoU at or above the threshold; if none
qualifies it may match an *ignored* ground truth (a crowd region, or a box
outside the current area range) and is then excluded from both TP and FP
counts. Crowd regions use intersection-over-prediction-area and may absorb
any number of predictions. All ten IoU thresholds are processed at once as
a `(T, M)` "taken" table.

## Average precision

`average_precision` sorts by score, accumulates TP and FP, makes precision
monotonically non-increasing from the right, and samples it at 101 recall
points. A worked case: predictions ranked TP, FP, TP against two ground
truths give precision `1, 0.5, 0.667` at recall `0.5, 0.5, 1`; the
envelope is `1, 0.667, 0.667`, so 51 recall points sample 1 and 50 sample
0.667, and AP is `(51 + 50·2/3) / 101`.

`MetricAccumulator` keeps per-class stores for the four COCO area ranges,
caps detections at 100 per image **and class**, uses the annotation's own
area for the size buckets, and reduces to `DetectionMetrics` (mAP50-95,
mAP50, mAP75, small/medium/large, per-class).

`tests/eval/test_pycocotools_parity.py` builds random datasets with crowd
regions, jittered and mislabelled detections and noise boxes, runs both
evaluators, and asserts all six summary numbers agree to 1e-6.

## Evaluator

`Evaluator(model, nc, EvalConfig(conf=0.001, iou=0.7, max_det=300, nms=True))`
runs the model on batches, applies `postprocess` (every `(anchor, class)`
pair above `conf` is a candidate, then class-aware NMS or a plain top-k
for the NMS-free head), undoes the letterbox and feeds the accumulator. A
`gt_lookup` callable supplies ground truth with crowd flags and areas; the
COCO script builds it from the json.

The result on COCO val2017 with converted GELAN-C weights is in
[Benchmarks](../benchmarks.md).

Source: `src/yolali/eval/`. API: [yolali.eval](../api/eval.md).
