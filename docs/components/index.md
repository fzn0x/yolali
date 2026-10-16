# Components

yolali is organised by the boundaries a detector actually has. Each page
below is one boundary: what crosses it, in what shape, and where the code
is. Reading them in order follows one batch from pixels to mAP.

| Stage | Package | Crosses the boundary |
|---|---|---|
| Boxes, anchors, decoding, NMS | [`yolali.ops`](ops.md) | pure tensor functions, no parameters |
| Label assignment | [`yolali.assign`](assign.md) | predictions + targets -> per-anchor targets |
| Loss | [`yolali.loss`](loss.md) | `RawPrediction` + `Targets` -> scalar terms |
| Backbone, neck, head | [`yolali.nn`](model.md) | image -> `RawPrediction` |
| Data | [`yolali.data`](data.md) | files -> `Batch` |
| Training | [`yolali.train`](train.md) | loop, EMA, schedule, checkpoints, loggers |
| Evaluation | [`yolali.eval`](eval.md) | detections -> COCO-protocol mAP |
| Inference and export | [`yolali.infer`, `yolali.export`](infer-export.md) | images -> `Detections`; model -> ONNX / TorchScript |

Two conventions hold everywhere:

- Boxes are `xyxy` in **pixels** at every public boundary. Normalised and
  `cxcywh` forms exist only inside the readers and the conversion helpers.
- Feature pyramids are `list[Tensor]` at strides `(8, 16, 32)`, in that
  order, whatever backbone produced them.

The shapes written in docstrings use `B` (batch), `A` (anchors, summed
over levels), `M` (padded ground-truth boxes per image), `nc` (classes)
and `reg_max` (DFL bins; `0` means direct regression).
