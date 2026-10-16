# Build plan

Eight phases, each ending with: full test suite green, `docs/design.md`
updated, and a short written summary with *measured* numbers only. The
design is in [`design.md`](design.md). Gates are the acceptance criteria; a
phase is not done until its gate passes.

Hardware available for gates: one NVIDIA RTX 3060 12 GB (Windows 11),
Python 3.12, CUDA 12.8 wheels via uv.

## Phase 0: scaffolding — done

- Repo, license, NOTICE, `pyproject.toml`, CI skeleton, docs.
- **Gate:** `uv run pytest`, `ruff check`, `pyright` pass on the empty
  package. Awaiting approval before phase 1.

## Phase 1: box ops, assigners, losses — done

Files: `ops/boxes.py`, `ops/anchors.py`, `ops/nms.py`, `assign/tal.py`,
`assign/o2o.py`, `loss/*.py`, `types.py`.

Tests with hand-computed values:

- IoU/GIoU/CIoU on three fixed box pairs (worked in the test docstring).
- `make_anchors` for a 2-level `(4×4, 2×2)` toy pyramid.
- `batched_nms` on five boxes, two classes: exact kept indices.
- TAL on a 1-image, 2-GT, 16-anchor toy: exact `fg_mask`, `target_gt_idx`,
  normalised scores.
- O2O on the same toy: exactly one anchor per GT.
- `DetectionLoss` on a 1-anchor, 1-GT case where every term reduces to a
  closed form (BCE of a known logit, `1 - IoU` of two known boxes, DFL of a
  known fractional target).

**Gate 1:** suite green, ruff + pyright clean.

## Phase 2: GELAN backbone, DFL head, converter, equivalence — done (gate 2 passed)

Files: `nn/blocks.py`, `nn/backbone/gelan.py`, `nn/neck/gelan_pan.py`,
`nn/head/decode.py`, `nn/head/detect.py`, `nn/config.py`, `nn/model.py`,
`zoo/unpickle.py`, `zoo/yolov9.py`, `configs/models/gelan-{n,s,m,c}.yaml`,
`scripts/convert_yolov9.py`.

Steps:

1. Blocks with shape tests and `fuse()` equivalence tests.
2. GELAN-C graph; parameter count must equal the reference (25.3 M for
   gelan-c; the exact integer is asserted once the converter runs).
3. Stub unpickler; converter table; round-trip test on key sets.
4. Reference outputs: run the GPL reference repo in a *separate* checkout,
   outside this repository, to dump `(1,3,640,640)` → three head outputs
   to `tests/fixtures/gelan_c_ref.npz` and `yolov9_c_conv_ref.npz` with
   the input seed recorded. Commit fixtures.
5. Equivalence test: `max |ours - ref| < 1e-4`.

**Gate 2:** equivalence test passes for `gelan-c.pt` and
`yolov9-c-converted.pt`.

## Phase 3: data pipeline — done

Files: `data/formats/*.py`, `data/cache.py`, `data/transforms/*.py`,
`data/dataset.py`, `data/collate.py`, `scripts/download_datasets.py`.

Tests: each transform preserves class counts and keeps boxes inside the
canvas; `Letterbox` inverse maps a box back within 0.5 px; cache rebuilds
when an mtime changes and not otherwise; COCO json and YOLO txt readers
agree on COCO128 (converted both ways).

**Gate:** suite green; COCO128 loads and a 16-image mosaic batch renders
to `runs/debug/` for eyeballing (image committed to docs, not to tests).

## Phase 4: Trainer and evaluator; gate 3 — in progress

Files: `train/*.py`, `eval/*.py`, `scripts/train.py`, `scripts/val.py`.

Steps:

1. `eval/map.py` with hand cases, then pycocotools cross-check to `1e-6`.
2. `scripts/val.py` on COCO val2017 with converted GELAN-C weights, classic
   head + NMS (`conf=0.001, iou=0.7, max_det=300`, letterbox 640, rect).
3. `Trainer` with every feature flag; 2-epoch resume bitwise test; DDP
   smoke on a single GPU with `torchrun --nproc_per_node=1`.
4. COCO128 50-epoch run with `gelan-n`, DFL head (dual head arrives in
   phase 5).

**Gate 3:** COCO val2017 mAP50-95 through our evaluator = 52.3 ± 0.3.
**Gate 4 (part 1):** COCO128 smoothed loss monotone decreasing, mAP50 > 0.3.
Both recorded in `BENCHMARKS.md` with command, commit, hardware, time.

## Phase 5: NMS-free dual head and direct regression; gate 6

Files: `nn/head/dual.py`, `loss/dual.py`, direct decode in `decode.py`,
`configs/models/*` switched to `head: dual, reg_max: 0` for N/S/M.

Steps:

1. Dual head + dual loss with toy tests (o2o branch picks exactly one
   anchor per GT; eval mode returns only o2o).
2. VOC 30 epochs, `gelan-n`, classic head + NMS → `BENCHMARKS.md`.
3. VOC 30 epochs, `gelan-n`, dual head, NMS-free inference → `BENCHMARKS.md`.

**Gate 4 (part 2):** VOC mAP50 > 0.6.
**Gate 6:** NMS-free within 1.0 mAP50 of NMS at the same budget. If it
fails, investigate the assigner (metric normalisation, o2o tie-breaking)
before anything else; document findings in `design.md`.

## Phase 6: ViT backbone with Spatial Tuning Adapter

Files: `nn/backbone/vit.py`, `nn/backbone/sta.py`,
`configs/models/vit-{s,m}.yaml`.

Tests: output strides and channels for 640 and 512 inputs; adapter
parameter count; frozen-backbone flag leaves ViT grads `None`.

**Gate:** suite green; a VOC 30-epoch `vit-s` run recorded in
`BENCHMARKS.md` (no numeric target set in the brief; the number is
reported as measured).

## Phase 7: export and inference API

Files: `infer/*.py`, `export/*.py`, `scripts/detect.py`,
`scripts/export.py`, `YolaliModel.export`.

Tests: `Detector.predict` on CPU with a tensor, an ndarray, and a path;
batched call returns one `Detections` per input; ONNX Runtime parity
`< 1e-4` for the dual direct-regression head; TorchScript parity.

**Gate 5:** ONNX parity test passes.

## Phase 8: CI, docs site, branding, benchmarks

- CI already runs lint + types + tests on push; nightly COCO128 smoke gets
  wired to `scripts/train.py`.
- `docs/` becomes an MkDocs Material site deployed to GitHub Pages:
  quickstart, concepts (one page per component), API reference via
  mkdocstrings, design doc, benchmarks.
- Branding: a logo and banner (SVG + PNG) used identically in the README
  and the site header; a CLI usage screenshot of `scripts/detect.py`
  produced from a real run.
- `BENCHMARKS.md` finalised; `CHANGELOG.md` cut to `0.1.0`; PyPI
  trusted-publishing workflow on tag.

**Gate:** site builds in CI; all six verification gates recorded.

## Summary format after each phase

```
Phase N done. Commits: <first>..<last>
Changed: <one line per module group>
Measured: <metric> = <value> (<command>, <hardware>, <time>) | none this phase
Design doc: <sections updated>
Next: Phase N+1
```
