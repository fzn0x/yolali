# API reference

Generated from the docstrings in `src/yolali/`. Every public function and
class states the shapes of the tensors it takes and returns.

| Package | Contents |
|---|---|
| [yolali.ops](ops.md) | box conversions, IoU family, anchors, decoding, NMS |
| [yolali.assign](assign.md) | `TaskAlignedAssigner`, `OneToOneAssigner`, `Assignment` |
| [yolali.loss](loss.md) | `LossConfig`, `DetectionLoss`, `DualLoss`, loss terms |
| [yolali.nn](nn.md) | blocks, backbones, necks, heads, `ModelConfig`, `YolaliModel` |
| [yolali.data](data.md) | readers, cache, transforms, dataset, collate |
| [yolali.train](train.md) | `TrainConfig`, `Trainer`, EMA, schedule, checkpoints, loggers |
| [yolali.eval](eval.md) | matching, AP, `MetricAccumulator`, `Evaluator` |
| [yolali.infer](infer.md) | `Detector`, `Detections`, preprocessing, drawing |
| [yolali.export](export.md) | ONNX and TorchScript export |
| [yolali.zoo](zoo.md) | checkpoint conversion |
