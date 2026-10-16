# yolali design

> A YOLO you can actually read.

This document is the contract between the code and its readers. Every
component boundary below has a "why". When the implementation and this file
disagree, one of them is wrong and the fix is a commit that changes both.

Status: **v0, phase 3 complete** (ops, assign, loss, GELAN model, converter, data pipeline; gate 2 passed, see `BENCHMARKS.md`). Sections marked *(phase N)* describe what
will exist after that phase of [`plan.md`](plan.md). Nothing here claims a
measured number; those live only in `BENCHMARKS.md`.

---

## 1. Goals and non-goals

**Goals**

1. A detector a careful reader understands in one sitting. Each file does one
   thing, each function fits on a screen, and every tensor that crosses a
   boundary has its shape written down.
2. 2026-current architecture: NMS-free end-to-end inference by default, DFL
   optional, CNN *and* ViT backbones behind one interface.
3. Verifiable correctness before any training: a weight converter and an
   equivalence test against the official GELAN-C checkpoint, then a COCO
   val2017 reproduction through our own evaluator.
4. Library, not framework. Plain `nn.Module`s and frozen dataclasses. The
   user composes; we do not own `main()`.

**Non-goals**

- Matching the throughput of hand-fused CUDA kernels.
- Segmentation, pose, OBB, tracking. The head interface leaves room; we do
  not build them.
- A config-driven "do everything from YAML" system. YAML describes *model
  shape* only. Training is configured by a dataclass you construct in Python.

---

## 2. Package layout

```
src/yolali/
├── __init__.py          public re-exports: Detector, Detections, load_model, LossConfig, ...
├── types.py             shared aliases and small frozen dataclasses (Boxes, Sample, Batch)
├── ops/                 pure tensor functions, no parameters, no state
│   ├── boxes.py         format conversions, IoU family, ltrb <-> xyxy
│   ├── anchors.py       anchor points + stride tensor from feature shapes
│   ├── decode.py        DFL expectation / direct softplus -> xyxy pixels (shared by loss, infer, export)
│   └── nms.py           batched class-aware NMS
├── assign/              label assignment: predictions -> targets
│   ├── tal.py           TaskAlignedAssigner (one-to-many)
│   └── o2o.py           OneToOneAssigner (same metric, top-1)
├── loss/
│   ├── config.py        LossConfig (frozen dataclass)
│   ├── cls.py           BCE and varifocal classification losses
│   ├── box.py           IoU loss (CIoU / GIoU), DFL term
│   ├── detection.py     DetectionLoss: one head, one assigner, one LossConfig
│   └── dual.py          DualLoss: o2m branch + o2o branch
├── nn/
│   ├── blocks.py        Conv, RepConv, RepBottleneck, RepCSP, RepCSPELAN, ADown, SPPELAN
│   ├── backbone/
│   │   ├── gelan.py     GELAN backbone -> [P3, P4, P5]
│   │   ├── vit.py       timm/HF ViT wrapper -> single 1/16 map
│   │   └── sta.py       SpatialTuningAdapter: 1/16 map -> [P3, P4, P5]
│   ├── neck/
│   │   └── gelan_pan.py GELAN PAN-FPN neck
│   ├── head/
│   │   ├── detect.py    DetectHead (one set of branches; DFL or direct)
│   │   └── dual.py      DualDetectHead (o2m + o2o branches; picks o2o at eval)
│   ├── config.py        ModelConfig + YAML loader + width/depth scaling rules
│   └── model.py         YolaliModel: backbone -> neck -> head; export()
├── data/
│   ├── formats/         yolo_txt.py, coco_json.py -> list[Sample]
│   ├── cache.py         label cache with content-hash invalidation
│   ├── transforms/      letterbox, mosaic, mixup, hsv, perspective, flip, compose
│   ├── dataset.py       DetectionDataset(samples, transform)
│   └── collate.py       collate_fn + RectBatchSampler
├── train/
│   ├── config.py        TrainConfig (frozen dataclass)
│   ├── ema.py           ModelEMA
│   ├── schedule.py      warmup + cosine LR, multi-scale schedule
│   ├── checkpoint.py    save/load (model, ema, optimizer, scaler, rng, epoch)
│   ├── loggers/         Logger protocol; csv.py, tensorboard.py, wandb.py
│   └── trainer.py       Trainer
├── eval/
│   ├── matching.py      per-image TP/FP matching at IoU thresholds
│   ├── map.py           COCO-style AP (101-point, area ranges, maxDets)
│   └── evaluator.py     Evaluator: dataloader -> Metrics
├── infer/
│   ├── detections.py    Detections dataclass
│   ├── preprocess.py    letterbox for inference + inverse mapping
│   └── detector.py      Detector.predict
├── export/
│   ├── onnx.py
│   └── torchscript.py
└── zoo/
    ├── unpickle.py      stub-class unpickler for foreign checkpoints
    └── yolov9.py        key remapping official GELAN/YOLOv9 -> yolali
```

Scripts (CLI wrappers, `argparse` lives here and only here):

```
scripts/train.py  scripts/val.py  scripts/detect.py  scripts/export.py
scripts/convert_yolov9.py  scripts/download_datasets.py
```

**Why this split.** `ops` has no parameters and no state, so it can be tested
with hand-computed numbers and reused by the loss, the evaluator, and
inference without import cycles. `assign` and `loss` know nothing about
`nn`: they consume raw tensors in a documented layout. `nn` knows nothing
about training. `train` composes all of them.

---

## 3. Tensor conventions

These hold everywhere unless a docstring says otherwise.

| Name | Shape | Meaning |
|---|---|---|
| image | `(B, 3, H, W)` float32 in `[0, 1]`, RGB | after letterbox, `H` and `W` multiples of 32 |
| features | `list[Tensor]` of `(B, C_i, H/s_i, W/s_i)` | `s = (8, 16, 32)`, in that order |
| anchor points | `(A, 2)` float | centre of each cell in **pixels**, `A = sum(H_i * W_i)` |
| strides | `(A, 1)` float | stride of the level each anchor belongs to |
| raw box | `(B, A, 4 * reg_max)` or `(B, A, 4)` | DFL logits or direct `ltrb` in stride units |
| decoded box | `(B, A, 4)` xyxy **pixels** | after `decode.py` |
| cls logits | `(B, A, nc)` | pre-sigmoid |
| targets | `Targets(boxes (B, M, 4) xyxy px, classes (B, M), mask (B, M))` | frozen dataclass in `types.py`; `M` = max GT per batch |

Boxes are `xyxy` in pixels at every public boundary. `cxcywh` and normalised
formats exist only inside `data/formats/*` (reading) and `ops/boxes.py`
(conversion). **Why:** one format eliminates a whole category of bugs, and
pixels make hand-computed tests readable.

---

## 4. Components

### 4.1 `ops` *(phase 1)*

`boxes.py`

- `xyxy_to_cxcywh`, `cxcywh_to_xyxy`, `ltrb_to_xyxy(anchors, ltrb)`,
  `xyxy_to_ltrb(anchors, xyxy)`.
- `box_iou(a, b) -> (N, M)`, `elementwise_iou(a, b) -> (N,)`, both thin
  wrappers over one `broadcast_iou` so the formulas exist once. Variants
  `"iou" | "giou" | "ciou"`. CIoU follows Zheng et al. 2020; the aspect term
  uses `atan` on `w/h` with `eps=1e-7` inside every division and a
  gradient-free `alpha`.

`decode.py`

- `dfl_expectation(raw, reg_max)`, `direct_distances(raw)` (softplus) and
  `decode_boxes(raw, anchors, strides, reg_max) -> (B, A, 4)` xyxy pixels.
  Lives in `ops` rather than the head so `loss` never imports `nn`.

`anchors.py`

- `make_anchors(feat_shapes, strides, offset=0.5) -> (anchors (A,2), strides (A,1))`.

`nms.py`

- `batched_nms(boxes, scores, classes, iou_thr, max_det)` using
  `torchvision.ops.batched_nms` with the class-offset trick. Exposed as a
  function so the evaluator and `Detector` share one implementation.

All of these are tested against values worked out by hand in the test file
(three-box IoU matrices, a known CIoU for two offset squares, etc.).

### 4.2 `assign` *(phase 1)*

**TaskAlignedAssigner** (`tal.py`). Inputs: decoded pred boxes `(B, A, 4)`,
pred scores `(B, A, nc)` (post-sigmoid), anchors `(A, 2)`, targets, mask.
Output: `target_boxes (B, A, 4)`, `target_scores (B, A, nc)`,
`fg_mask (B, A)` bool, `target_gt_idx (B, A)`.

Algorithm, written to be read top to bottom:

1. `in_gt = anchor inside GT box` → `(B, M, A)` bool.
2. `metric = score[gt_cls]^alpha * iou^beta` → `(B, M, A)`, masked by `in_gt`.
3. For each GT pick `topk` anchors by metric → candidate mask.
4. If an anchor is a candidate for several GTs, keep the GT with the highest
   IoU (not metric). *Why IoU:* it is the choice the YOLOv8/v9 lineage
   makes, and it keeps the tie-break independent of classification quality.
5. `target_scores = one_hot(cls) * (metric / metric.max_per_gt * iou.max_per_gt)`
   (the "normalised alignment metric" soft target).

Defaults: `topk=10, alpha=0.5, beta=6.0`.

**OneToOneAssigner** (`o2o.py`). The YOLOv10 "consistent dual assignment":
the *same* metric with `topk=1`. Implemented as `TaskAlignedAssigner(topk=1)`
plus a guarantee that each GT gets at most one anchor. It is a separate
class so the loss reads as "o2m assigner, o2o assigner" rather than a flag.

### 4.3 `loss` *(phase 1, extended in phase 5)*

```python
@dataclass(frozen=True)
class LossConfig:
    box_weight: float = 7.5
    cls_weight: float = 0.5
    dfl_weight: float = 1.5          # ignored when reg_max == 0
    iou_kind: Literal["ciou", "giou"] = "ciou"
    cls_loss: Literal["bce", "vfl"] = "bce"
    reg_max: int = 16                # 0 => direct ltrb regression
    tal_topk: int = 10
    tal_alpha: float = 0.5
    tal_beta: float = 6.0
    o2o_weight: float = 1.0          # weight of the one-to-one branch in DualLoss
    vfl_alpha: float = 0.75
    vfl_gamma: float = 2.0
```

`DetectionLoss(cfg, nc, assigner=None)` takes **raw head outputs**, the
anchors and strides they were produced on, and `Targets`; it returns a
`LossOutput(total, box, cls, dfl, num_fg)`. It owns one assigner. It never
touches the model.

- Classification: BCE on `target_scores` (soft) or varifocal
  (`alpha=0.75, gamma=2.0`).
- Box: `1 - IoU_variant` on fg anchors, weighted by `target_scores.sum(-1)`.
- DFL (only if `reg_max > 0`): cross-entropy against the two integer bins
  bracketing each `ltrb` target in stride units, clamped to `[0, reg_max-1-0.01]`.
- Normalisation: all three terms divided by `target_scores.sum()` clamped to
  `>= 1`, then the weighted sum is multiplied by the batch size (YOLOv8/v9
  convention) so the default weights and learning rate transfer across
  batch sizes. Per-term values reported for logging are *before* that
  multiplication.
- When no anchor is positive, box and DFL terms are `raw.sum() * 0` so the
  graph stays connected and DDP never sees an unused parameter.

`DualLoss(cfg, nc)` holds two `DetectionLoss`es: one with the
`TaskAlignedAssigner`, one with the `OneToOneAssigner`. Total is
`o2m + o2o_weight * o2o`. Per-branch terms are reported separately so a
training log shows both.

**Why decoupled:** the loss can be unit-tested with synthetic predictions
and hand-computed expectations, and a user can plug in a different assigner
without touching the model.

### 4.4 `nn.blocks` *(phase 2)*

Reimplemented from the YOLOv9 paper and config files (Wang, Yeh, Liao 2024,
arXiv:2402.13616). Every block's docstring cites the paper and the
corresponding name in the reference repo so a reader can cross-check.

| Block | What it is |
|---|---|
| `Conv` | Conv2d → BatchNorm2d(eps 1e-3, momentum 0.03, as in the reference) → SiLU. `fuse()` folds BN for inference. |
| `RepConv` | 3×3 and 1×1 parallel conv-BN branches summed, SiLU; `fuse()` collapses to one 3×3. Training-time form only, no identity branch (as in the reference). |
| `RepBottleneck` | `RepConv(3×3)` → `Conv(3×3)` with optional residual. |
| `RepCSP` | CSP split: `n` bottlenecks on one branch, 1×1 on the other, concat, 1×1. |
| `RepCSPELAN` | GELAN unit: 1×1 to `2c`, split into two, chain two `(RepCSP → Conv3×3)` stages, concat all four, 1×1 out. |
| `ADown` | Downsample: avgpool-2 → split → (Conv3×3 s2) ∥ (maxpool-3 s2 → Conv1×1), concat. |
| `SPPELAN` | Conv1×1 → three cascaded maxpool-5 → concat(4) → Conv1×1. |

Channel arguments are explicit `(c_in, c_out, c_hidden, n)`; no `gw`/`gd`
magic inside the block. Scaling lives in `nn/config.py`.

### 4.5 `nn.backbone.gelan` and `nn.neck.gelan_pan` *(phase 2)*

The GELAN-C topology from the paper, expressed as an explicit sequence of
named attributes rather than a parsed list, so `print(model)` is a
readable network diagram:

```
stem:  Conv(3,  64, 3, s2)      # P1  /2
       Conv(64, 128, 3, s2)     # P2  /4
       RepCSPELAN(128, 256, 128, 64, n=1)
down3: ADown(256, 256)          # P3  /8
       RepCSPELAN(256, 512, 256, 128, n=1)      -> out P3
down4: ADown(512, 512)          # P4  /16
       RepCSPELAN(512, 512, 512, 256, n=1)      -> out P4
down5: ADown(512, 512)          # P5  /32
       RepCSPELAN(512, 512, 512, 256, n=1)      -> out P5
```

Neck (PAN): `SPPELAN` on P5, top-down with nearest upsample + concat +
`RepCSPELAN`, bottom-up with `ADown` + concat + `RepCSPELAN`. Outputs three
maps at strides 8/16/32 with channels `(256, 512, 512)` for the C size.

Width and depth multipliers apply to every `c_out`/`c_hidden`/`n` through
one function `scale(c, width)` = `make_divisible(c * width, 8)`, `n =
max(round(n * depth), 1)`. N/S/M configs in `configs/models/` are *our*
scalings of this topology; they are not claimed to match any official
`gelan-t/s/m` checkpoint (those use different blocks).

**PGI (auxiliary reversible branch).** The paper's Programmable Gradient
Information adds a training-only auxiliary branch that is removed at
inference. Equivalence testing targets the inference graph, which is
identical for `gelan-c.pt` and `yolov9-c-converted.pt`. PGI is therefore
*not* in scope for the gates. It is listed under "Future work" with a note
on where it would plug in (a second head consuming backbone features via
`CBLinear`/`CBFuse`).

### 4.6 `nn.head` *(phase 2, dual in phase 5)*

`DetectHead(ch: tuple[int,...], nc, reg_max)` per level `i`:

```
box branch: Conv(ch_i, c2, 3) → Conv(c2, c2, 3, groups=g) → Conv2d(c2, 4*reg_max or 4, 1, groups=g)
cls branch: Conv(ch_i, c3, 3) → Conv(c3, c3, 3)           → Conv2d(c3, nc, 1)
```

with `c2 = make_divisible(max(ch[0] // 4, 4 * reg_max, 16), 4)`,
`c3 = max(ch[0], min(2 * nc, 128))`, and `g = 4` for DFL heads (matching the
reference `DDetect` weight shapes, which the equivalence test reproduces) or
`g = 1` for the direct-regression head, whose four output channels would
otherwise each see a quarter of the features. The head returns
`(box_raw (B, A, 4*reg_max | 4), cls_logits (B, A, nc))` with levels
flattened and concatenated in stride order. Decoding is *not* done here.

See `ops/decode.py` (4.1): the head calls `decode_boxes` and nothing else.

`DualDetectHead` holds two `DetectHead`s on the same neck features: `o2m`
and `o2o`. During training it returns both. In `eval()` mode it returns
only `o2o`. The o2o branch is a full copy (not shared convs). *Why:* YOLOv10
shares nothing but the neck either; sharing would couple the two
assignment regimes and muddy the gate-6 comparison.

Inference postprocess for the o2o path: `scores = sigmoid(cls)`, take
`max_det` anchors by max class score via `topk`, threshold at `conf`. No
NMS. For the o2m path: same, then `batched_nms`.

### 4.7 `nn.config` and `nn.model` *(phase 2)*

YAML is deliberately flat:

```yaml
# configs/models/gelan-n.yaml
name: gelan-n
nc: 80
backbone: { type: gelan, width: 0.25, depth: 0.33 }
neck:     { type: gelan_pan, width: 0.25, depth: 0.33 }
head:     { type: dual, reg_max: 0 }      # NMS-free, direct regression
```

```yaml
# configs/models/gelan-c.yaml   (equivalence target)
name: gelan-c
nc: 80
backbone: { type: gelan, width: 1.0, depth: 1.0 }
neck:     { type: gelan_pan, width: 1.0, depth: 1.0 }
head:     { type: detect, reg_max: 16 } # classic head + NMS
```

```yaml
# configs/models/vit-s.yaml
name: vit-s
nc: 80
backbone: { type: vit, model: vit_small_patch16_dinov2, img_size: 640 }
neck:     { type: gelan_pan, width: 0.5, depth: 0.33 }
head:     { type: dual, reg_max: 0 }
```

`ModelConfig.from_yaml` parses into frozen dataclasses with a small registry
(`dict[str, type[Backbone]]`) that users can extend by passing their own
mapping. No import-time side effects, no decorators that register globally.

`YolaliModel(backbone, neck, head, cfg)` is three attributes and a `forward`
returning a `RawPrediction(box_raw, cls_logits, anchors, strides)`;
`decode()` turns that into pixel boxes and probabilities. It also has
`fuse()` (BN folding) and, from phase 7, `export(format, path, **kw)`.
`Backbone` and `Neck` in `nn/interfaces.py` are thin `nn.Module` base
classes that spell out the contract (`out_channels`, `strides`, `forward`).

### 4.8 `zoo` *(phase 2)*

Official checkpoints are pickled `nn.Module` objects that reference the
reference repo's module paths. We must not import that code. `unpickle.py`
installs a custom `Unpickler.find_class` that maps any `models.*` class to
a `StubModule` whose `__setstate__` records `_modules`, `_parameters`, and
`_buffers`. Walking that tree yields a plain `dict[str, Tensor]` with the
reference's key names (`model.0.conv.weight`, ...).

`yolov9.py` holds one explicit table `REFERENCE_INDEX -> yolali attribute`
for the GELAN-C graph (e.g. `model.3` → `backbone.stage3.elan`) and remaps
parameter names mechanically. The table is data, so a reader can audit it
line by line. It raises on any unmatched reference key; `load_state_dict(strict=True)`
catches the other direction. `tests/fixtures/gelan_c_reference_shapes.json`
(the reference `state_dict` key → shape table) lets CI verify the mapping
without the 51 MB checkpoints.

Equivalence test (gate 2): load `gelan-c.pt`, convert, run both on a fixed
`(1, 3, 640, 640)` input, compare the three raw head outputs. The reference
forward is obtained by running the reference repo in a subprocess from a
*separate* checkout, saving outputs to `.npz`. The 320 px `.npz` dumps
(~2 MB each) are committed as test fixtures so CI never needs the GPL code;
640 px dumps are kept locally. The threshold is `max |a - b| < 1e-4` in
float32 with `fuse()` disabled. Measured results are in `BENCHMARKS.md`.

A detail worth recording: the reference sets `BatchNorm2d(eps=1e-3,
momentum=0.03)` on every layer at build time. With PyTorch's default `eps`
the max abs diff was ~10; with the reference values it is < 1e-4.

### 4.9 `data` *(phase 3)*

```python
@dataclass
class Sample:
    image: np.ndarray        # (H, W, 3) uint8 RGB
    boxes: np.ndarray        # (N, 4) float32 xyxy pixels
    classes: np.ndarray      # (N,) int64
    path: str
    orig_hw: tuple[int, int] # before any transform
    scale: float             # cumulative resize factor
    pad: tuple[float, float] # (left, top) letterbox padding
```

Readers (`formats/yolo_txt.py`, `formats/coco_json.py`) return
`list[LabelRecord]` (path, size, boxes, classes, crowd flags) *without*
loading pixels. `cache.py` stores that list in a `.cache` file keyed by
`sha1(sorted(path, size, mtime_ns) + reader_version)`. Any mismatch rebuilds;
a corrupt cache is silently rebuilt too.

Transforms are callables `(Sample, np.random.Generator) -> Sample` composed
by `Compose`. **Every random draw comes from the generator argument**, and
the dataset derives one generator per `(seed, epoch, index)`, so a sample is
reproducible and two workers never share state. The transforms that need
other images (`Mosaic`, `MixUp`) receive a `sampler: Callable[[Generator],
Sample]` at construction, so the dataset stays a plain `__getitem__`.
`Letterbox(size, auto, stride)` records `scale` and `pad` on the sample;
`undo_letterbox` inverts them.

Default training pipeline (`default_train_transforms`):
`Mosaic(2s×2s canvas) → RandomPerspective(border=-s/2 → s×s) → Letterbox →
MixUp(partner = same geometry pipeline) → HSV → flips`. Validation is
`Letterbox` only. Tensors are made in `collate_samples`, not in a transform,
so a `Sample` is always a numpy object a reader can print.

`RectBatchSampler` sorts images by aspect ratio and computes one `(H, W)`
per batch (multiples of `stride`); the dataset applies it through
`rect_shapes` after the transform pipeline. Used for validation.

### 4.10 `train` *(phase 4)*

```python
@dataclass(frozen=True)
class TrainConfig:
    epochs: int
    batch_size: int
    img_size: int = 640
    lr: float = 0.01
    momentum: float = 0.937
    weight_decay: float = 5e-4
    warmup_epochs: float = 3.0
    cosine: bool = True
    amp: bool = True                 # bf16 if supported, else fp16 + GradScaler
    ema: bool = True
    ema_decay: float = 0.9999
    close_mosaic: int = 10
    accumulate: int = 1
    multi_scale: tuple[int, int] | None = None   # (min, max) multiples of 32
    compile: bool = False
    seed: int = 0
    out_dir: Path = Path("runs/exp")
```

`Trainer(model, loss, train_loader, val_loader, cfg, loggers=(), evaluator=None)`
has `fit()` and `resume(path)`. Each feature is an `if cfg.flag:` at one
place in the loop, not a strategy hierarchy. DDP is detected from
`torchrun` environment variables; the user never passes a rank.

`Logger` is a `Protocol` with `log_scalars(step, values)`, `log_config(cfg)`,
`close()`. `CsvLogger` is always available; TensorBoard and W&B are optional
extras that raise a clear `ImportError` message.

Checkpoint = `{model, ema, optimizer, scaler, rng (torch, cuda, numpy,
python), epoch, cfg, version}`. `resume` restores all of it so a resumed run
is bitwise identical to an uninterrupted one on the same hardware (tested
for two short epochs).

### 4.11 `eval` *(phase 4)*

Our own COCO-protocol mAP, not a pycocotools wrapper:

- `matching.py`: for one image, greedy match predictions (sorted by score)
  to GTs at each IoU threshold in `linspace(0.5, 0.95, 10)`; returns a
  `(N_pred, 10)` TP matrix.
- `map.py`: accumulate per class, compute precision/recall, 101-point
  interpolated AP, area ranges (`all/small/medium/large`), `maxDets=100`.

Tested against hand-computed cases (one class, three predictions, two GTs)
and, under the `coco` extra, against `pycocotools` on a random synthetic
set to `1e-6`. **Why both:** the hand cases make the algorithm legible; the
pycocotools cross-check is what gate 3 depends on.

### 4.12 `infer` *(phase 7)*

```python
@dataclass(frozen=True)
class Detections:
    xyxy: Tensor    # (N, 4) float32 pixels in the ORIGINAL image
    conf: Tensor    # (N,) float32
    cls: Tensor     # (N,) int64
    names: tuple[str, ...] | None = None
```

`Detector(model, device="auto", half=False)` accepts paths, `np.ndarray`
HWC uint8, PIL images, or a `(B,3,H,W)` tensor. It letterboxes, runs the
model, undoes letterbox, and returns `list[Detections]`. `conf`, `iou` (NMS
path only), and `max_det` are `predict()` arguments.

### 4.13 `export` *(phase 7)*

`export_onnx(model, path, img_size, opset=17, dynamic_batch=True)` wraps the
model in a tiny `ExportWrapper` whose `forward` returns `(B, max_det, 6)` =
`[x1, y1, x2, y2, conf, cls]` for the o2o head (decode + sigmoid + topk,
all standard ops) or `(B, A, 4 + nc)` for the o2m head (NMS left to the
caller). We use the TorchScript-based exporter (`dynamo=False`) because its
op coverage for this graph is complete and stable across 2.4–2.x.

Parity test: ONNX Runtime CPU vs PyTorch CPU on the same input,
`max |a - b| < 1e-4` for the direct-regression dual head.

### 4.14 ViT backbone with Spatial Tuning Adapter *(phase 6)*

`ViTBackbone(model_name, img_size)` loads a DINOv2 or DINOv3 small variant
through `timm` (`vit_small_patch14_dinov2` with `patch_size=16` override,
which timm resamples; or the HF DINOv3 ViT-S/16 when available) and
returns the patch-token grid reshaped to `(B, C, H/16, W/16)`.

`SpatialTuningAdapter(c_in, out_channels=(c3, c4, c5))` (after DEIMv2,
Huang et al. 2025):

```
P4 = Conv1x1(c_in, c4)(x16)
P3 = Conv3x3(c3)(ConvTranspose2d(c_in, c3, k=2, s=2)(x16))
P5 = Conv3x3 s2 (c_in, c5)(x16)
```

Each branch is followed by one `RepCSPELAN`-free `Conv` so the adapter stays
tiny. The ViT weights may be frozen (`freeze_backbone=True`) for the first
`k` epochs via a `Trainer` hook. See Deviations D4.

License note: DINOv2 weights are Apache-2.0; DINOv3 weights carry Meta's
DINOv3 license. The default config uses DINOv2.

---

## 5. Deviations and ambiguity resolutions

Each entry: what the source says (or does not say), what we chose, why.

- **D1. TAL conflict resolution by IoU, not metric.** The TOOD paper is
  explicit about the alignment metric but not about multi-GT conflicts.
  We keep the GT with the highest IoU, matching the YOLOv8/v9 lineage.
- **D2. Direct regression activation.** YOLO26 release notes say DFL was
  removed in favour of direct regression but do not specify how
  non-negativity is enforced. We use `softplus` on raw `ltrb` in stride
  units: smooth, standard ONNX op, no dead gradient at zero (unlike
  `relu`), no overflow (unlike `exp`). Loss is computed on decoded boxes,
  so the choice only affects optimisation dynamics.
- **D3. One-to-one branch is a full head copy.** YOLOv10 describes the o2o
  head as a separate branch; whether its stem convs are shared is an
  implementation detail. We share nothing beyond the neck.
- **D4. STA details.** DEIMv2 describes the Spatial Tuning Adapter at the
  block-diagram level. We implement the simplest pyramid: a 2× transposed
  conv for P3, a 1×1 for P4, a stride-2 conv for P5, each followed by one
  3×3 `Conv`. No cross-scale fusion inside the adapter; the neck does that.
- **D5. PGI not implemented for the gates.** See 4.5. The inference graph
  of `yolov9-c` is `gelan-c`, so equivalence and COCO reproduction do not
  need it.
- **D6. Model scaling.** Our N/S/M are width/depth scalings of the C
  topology. The official `gelan-t/s/m` use `AConv`/`ELAN1` blocks and are
  not targets.
- **D7. No ProgLoss / STAL / MuSGD.** YOLO26's additional training tricks
  are documented only in release notes. They are out of scope for v0 and
  listed under Future work.

---

## 6. Testing strategy

| Layer | Test style |
|---|---|
| `ops`, `assign`, `loss`, `eval.map` | Hand-computed expected values in the test file, plus property tests (symmetry, bounds). |
| `nn.blocks` | Output shapes, parameter counts vs. hand-derived formula, `fuse()` numerical equivalence. |
| `nn.model` + `zoo` | Gate 2 equivalence against committed `.npz` fixtures (`-m weights` for the live checkpoint). |
| `data` | Transform invariants (boxes stay inside image, class count preserved), cache invalidation on mtime change. |
| `train` | Two-step overfit on a synthetic batch; resume reproduces bitwise. |
| `export` | ORT vs PyTorch parity. |
| Gates 3, 4, 6 | `scripts/` runs recorded in `BENCHMARKS.md`; CI runs only the CPU smoke. |

---

## 7. Future work

- PGI auxiliary branch for training from scratch at C/E scale.
- ProgLoss, STAL, distillation from the ViT path to the CNN path (DEIMv2
  style).
- Segmentation and pose heads behind the same `Head` interface.
- `torch.export`-based ONNX path once its op coverage is stable.

---

## 8. Credits

GELAN, the PAN neck topology, the detect head layout and the checkpoint
format follow **Chien-Yao Wang, I-Hau Yeh and Hong-Yuan Mark Liao, "YOLOv9:
Learning What You Want to Learn Using Programmable Gradient Information",
arXiv:2402.13616**, repository github.com/WongKinYiu/yolov9 (GPL-3.0, read
but not copied). Dual assignment follows YOLOv10 (arXiv:2405.14458). The
Spatial Tuning Adapter follows DEIMv2 (arXiv:2509.20787). D-FINE
(arXiv:2410.13842) and RF-DETR informed the ViT-backbone path.
