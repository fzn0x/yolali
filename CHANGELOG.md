# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project
adheres to [Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added

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
