# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project
adheres to [Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added

- `yolali.train`: `TrainConfig` (one switch per feature), `Trainer` with AMP,
  EMA, warmup + cosine schedule, close-mosaic, gradient accumulation,
  multi-scale, `torch.compile`, DDP and bitwise resume; CSV, TensorBoard and
  W&B loggers; `scripts/train.py`.
- `yolali.eval`: COCO-protocol greedy matching with crowd and ignore
  handling, 101-point AP, area buckets, `MetricAccumulator`, `Evaluator`
  with letterbox undo; agrees with pycocotools to 1e-6; `scripts/val.py`.
- `yolali.nn.head.dual`: one-to-many / one-to-one dual head and `DualLoss`
  for NMS-free inference; N/S/M configs default to it with direct regression.
- `yolali.nn.backbone.vit`: DINOv2/DINOv3 ViT backbone through `timm` with
  patch-16 resampling and a Spatial Tuning Adapter (`sta.py`); `vit-s` and
  `vit-m` configs.
- `yolali.infer`: `Detector.predict` for tensors, arrays and paths, typed
  `Detections`, drawing helper; `scripts/detect.py`.
- `yolali.export`: ONNX (standard ops, dynamic shapes) and TorchScript with
  parity tests; `YolaliModel.export`; `scripts/export.py`.
- Public API re-exports at the package top level.
- MkDocs Material documentation site (datasheet, quickstart, one page per
  component, mkdocstrings API reference) deployed to GitHub Pages; logo,
  banner and block-diagram assets; nightly COCO128 smoke-train workflow;
  PyPI trusted-publishing workflow on tags.
- `yolali.data`: YOLO txt and COCO json readers, fingerprinted label cache,
  composable transforms (letterbox, mosaic, mixup, HSV, perspective, flips),
  `DetectionDataset`, `collate_samples`, `RectBatchSampler`;
  `scripts/download_datasets.py` and `scripts/debug_batch.py`.
- `yolali.nn`: GELAN blocks (`Conv`, `RepConv`, `RepCSP`, `RepCSPELAN`,
  `ADown`, `SPPELAN`) with BN/RepConv fusion, `GELANBackbone`, `GELANPAN`
  neck, `DetectHead` (DFL or direct regression), YAML `ModelConfig` and
  `YolaliModel`; N/S/M/C configs under `configs/models/`.
- `yolali.zoo`: stub-class unpickler and GELAN-C converter for official
  YOLOv9 checkpoints; forward-pass equivalence verified to < 1e-4
  (`BENCHMARKS.md`, gate 2).
- `yolali.ops`: box conversions, IoU/GIoU/CIoU, anchor points, DFL and
  direct-regression decoding, batched NMS; all tested against hand-computed
  values.
- `yolali.assign`: `TaskAlignedAssigner` (one-to-many) and
  `OneToOneAssigner` (YOLOv10-style) with worked toy-grid tests.
- `yolali.loss`: `LossConfig`, BCE / varifocal, IoU and DFL terms,
  `DetectionLoss` and `DualLoss` with closed-form single-anchor tests.
- `yolali.types.Targets`: padded ground-truth container.
- Repository scaffolding: Apache-2.0 license, `NOTICE`, `pyproject.toml`
  with uv / ruff / pyright configuration, CI skeleton.
- `docs/design.md`: component boundaries and the rationale for every
  architectural decision.
- `docs/plan.md`: the eight-phase build plan and its verification gates.
