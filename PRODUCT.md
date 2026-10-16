# Product

<!-- impeccable:product-schema 1 -->

## Platform

web

## Stack

MkDocs Material with mkdocstrings for the documentation site (confirmed by the
user on 2026-10-16); custom CSS and SVG assets carry the brand. Deployed to
GitHub Pages from GitHub Actions. The README on GitHub shares the same logo
and banner assets. Build path recorded in `.impeccable/config.json`: code-first.

## Users

Primary: ML engineers who read code. They open the repository to understand,
verify, and adapt a modern object detector in one sitting, and they judge the
project by code clarity and by measured numbers they can reproduce. They
arrive from GitHub (README first) or a search for a specific component
(assigner, loss, evaluator) and read docs beside an editor. Secondary, not
designed for: students meeting detection for the first time, and
practitioners comparing tools on throughput and mAP.

## Product Purpose

yolali is a clean-room PyTorch object detection library whose guiding
principle is "a YOLO you can actually read": every component is a plain
`nn.Module` or a typed dataclass, fully type-annotated, replaceable, and
tested against hand-computed values. It targets the 2026 state of the art
(NMS-free end-to-end inference, optional DFL, GELAN CNN and DINOv2/DINOv3
ViT backbones behind one interface). Success means a reader can trace a
box from raw head output to mAP without leaving the repository, and every
published number has a command, a commit, and hardware next to it.

## Positioning

The reference YOLO implementations are research code; yolali is a library a
reader understands in one sitting whose correctness is proven before any
training: the GELAN path reproduces the official YOLOv9 checkpoint forward
pass to < 1e-4 and reproduces the published COCO mAP (52.5 vs 52.3) through
its own evaluator, which itself matches pycocotools to 1e-6. Neighbouring
projects either hide behind a framework or ship unverified ports.

## Operating Context

- Repository: github.com/fzn0x/yolali, Apache-2.0, `uv`-managed, `src/`
  library and `scripts/` CLIs; CI runs lint, type check and CPU tests.
- Docs are read next to code in an editor and on GitHub; README is the
  front door, the site is the depth. API reference is generated from
  docstrings, which already state tensor shapes.
- Documents that are factual parts of the product: `docs/design.md`
  (component boundaries and deviations), `docs/plan.md` (phases and gates),
  `BENCHMARKS.md` (measured numbers), `CHANGELOG.md`, `CONTRIBUTING.md`.
- CLI usage (`scripts/detect.py`, `train.py`, `val.py`, `export.py`) is a
  real part of the experience and is to be shown as a genuine terminal
  screenshot in the README and on the site.

## Capabilities and Constraints

- Shipped: box ops, task-aligned and one-to-one assigners, DFL/CIoU/BCE/VFL
  losses, GELAN backbone + PAN neck, DFL and direct-regression heads, dual
  NMS-free head, DINOv2/DINOv3 ViT backbone with a Spatial Tuning Adapter,
  YOLO txt and COCO json data pipeline with composable augmentations,
  Trainer (AMP, EMA, warmup+cosine, close-mosaic, accumulation, multi-scale,
  compile, DDP, bitwise resume, CSV/TensorBoard/W&B loggers), COCO-protocol
  evaluator, `Detector.predict`, ONNX and TorchScript export.
- Terminology used in code and docs: P3/P4/P5 (strides 8/16/32), o2m/o2o
  (one-to-many / one-to-one), `RawPrediction`, `Targets`, `Detections`,
  "gate" (a verification criterion in docs/plan.md).
- Constraints: Apache-2.0 code; pretrained GELAN weights are GPL-3.0 from
  the YOLOv9 authors and are never bundled; no code from AGPL/GPL repos.
- Undecided: PyPI release name/version timing (package is `yolali`,
  version 0.0.1, pre-alpha).

## Brand Commitments

- Name: yolali (lower-case wordmark). Tagline: "A YOLO you can actually read."
- Visual direction pinned by the user on 2026-10-16: "technical blueprint"
  (monospace-led, grid and bounding-box motifs, one accent colour; reads
  like a well-typeset paper). Logo and banner must be the same assets in
  the README and on the site.
- Voice: plain, exact, no hype; numbers only when measured; credits
  WongKinYiu (YOLOv9/GELAN/PGI) as the primary architectural inspiration
  and names the YOLOv10, YOLO26, D-FINE, DEIMv2 and RF-DETR authors.

## Evidence on Hand

- `BENCHMARKS.md`: gate 2 equivalence table, gate 3 COCO val2017 numbers,
  gate 4a COCO128 curve; more runs land as they finish.
- `docs/img/coco128_mosaic_batch.jpg`: a real augmented training batch.
- `runs/detect/` output from `scripts/detect.py` on COCO128 images with the
  converted GELAN-C weights can be produced on demand for screenshots.
- No testimonials, users, downloads, or press. Do not fabricate any.

## Product Principles

1. Readability is the feature: every page should let a reader find the code
   that does the thing within seconds.
2. Nothing unmeasured is claimed; every number carries command, commit and
   hardware.
3. One interface, many parts: backbone, neck, head, loss, assigner are
   swappable, and the docs are organised by those boundaries.
4. Credit is explicit and license hygiene is visible.
5. The CLI is a first-class surface and is shown as it really runs.

## Accessibility & Inclusion

Standard web accessibility for a documentation site: keyboard navigation,
visible focus, contrast at WCAG AA, reduced-motion respected. No further
product-specific requirement was established.
