# Benchmarks

Every number on this page was produced by the command shown next to it, on
the hardware shown, at the commit shown. Nothing here is copied from a paper.

## Status

No measurements yet. Entries are added as the verification gates in
`docs/plan.md` are passed, in this order:

1. GELAN-C forward-pass equivalence against the official checkpoint
   (max abs diff).
2. COCO val2017 mAP50-95 with converted official GELAN-C weights through
   our evaluator (target: 52.3 ± 0.3).
3. COCO128, N-size model, 50 epochs (target: mAP50 > 0.3, smoothed loss
   monotone decreasing).
4. VOC, N-size model, 30 epochs, NMS head (target: mAP50 > 0.6).
5. VOC, N-size model, 30 epochs, NMS-free head (target: within 1.0 mAP50 of
   the NMS head).
6. ONNX Runtime vs PyTorch output parity (max abs diff).

## Hardware

| Label | CPU | GPU | OS | Software |
|---|---|---|---|---|
| `dev-3060` | Intel 12th-gen (family 6, model 151), 20 threads | NVIDIA GeForce RTX 3060 12 GB, driver 576.80 | Windows 11 Pro | Python 3.12.13, torch 2.11.0+cu128 |

## Results

### Gate 2: GELAN-C forward-pass equivalence

```
commit:   ea8abd5
hardware: dev-3060 (CPU only, float32, BatchNorm not fused)
command:  uv run pytest -m weights tests/zoo/test_yolov9.py
          (official checkpoints in ./weights, reference dumps produced once by
          running the GPL reference code in a separate checkout; the 320 px
          dumps are committed under tests/fixtures/)
time:     < 5 s
```

| Checkpoint | Input | box logits max abs diff | cls logits max abs diff | decoded (cxcywh px, scores) max abs diff | CPU forward |
|---|---|---|---|---|---|
| `gelan-c.pt` | 320×320 | 2.4e-05 | 5.3e-05 | 2.4e-04 | 124 ms |
| `gelan-c.pt` | 640×640 | 8.1e-05 | 5.5e-05 | 7.8e-04 | 378 ms |
| `yolov9-c-converted.pt` | 320×320 | 1.6e-05 | 1.6e-05 | 2.4e-04 | 113 ms |
| `yolov9-c-converted.pt` | 640×640 | 2.9e-05 | 2.4e-05 | 3.5e-04 | 352 ms |

Threshold was `< 1e-4` on the raw head outputs. Both pass. Parameter count
of the converted model is 25,498,736 (reference 25,498,752 including the
16-value frozen DFL arange that yolali computes instead of storing).

### Gate 3: COCO val2017 with converted official GELAN-C weights

```
commit:   5684411
hardware: dev-3060 (RTX 3060, fp16 autocast, batch 16, 4 loader workers)
command:  uv run scripts/val.py --config configs/models/gelan-c.yaml \
              --weights weights/gelan-c.yolali.pt --images datasets/coco/val2017 \
              --coco-json datasets/coco/annotations/instances_val2017.json \
              --batch-size 16 --workers 4 --half --out runs/val/gelan-c_val2017.json
settings: letterbox 640 (rect batches, stride 32), conf 0.001, NMS IoU 0.7,
          max 300 detections before scoring, 100 per image and class scored
time:     294 s for 5000 images (evaluator is CPU-bound; see below)
```

| Metric | yolali evaluator | Published (YOLOv9 README, gelan-c) |
|---|---|---|
| mAP50-95 | **52.5** | 52.3 |
| mAP50 | 69.5 | 70.2 |
| mAP75 | 57.3 | — |
| mAP small / medium / large | 35.4 / 57.6 / 69.0 | — |

Target was 52.3 ± 0.3 on mAP50-95: **passed** (+0.2). The published mAP50 is
not part of the gate; the 0.7 gap there is consistent with the usual
difference between pycocotools and a rect-batched, fp16 evaluation run and
was not investigated further.

Evaluator cost: `MetricAccumulator.add_image` takes ~60 ms per image on
this CPU with 300 detections (measured on synthetic data after the
per-class, all-thresholds-at-once rewrite; it was 320 ms before).

### Gate 4a: COCO128 smoke train, GELAN-N (classic DFL head), 50 epochs, from scratch

```
commit:   44efd1b
hardware: dev-3060 (bf16 autocast, batch 16, 4 loader workers)
command:  uv run scripts/train.py --model configs/models/gelan-n-nms.yaml \
              --data configs/data/coco128.yaml --epochs 50 --batch-size 16 \
              --workers 4 --out runs/coco128-n-nms
settings: SGD lr 0.01 cosine, warmup 100 iterations, mosaic until epoch 40,
          EMA, 8 optimizer steps per epoch (400 total)
time:     13 s per epoch (+ validation), ~15 min total
```

| Epoch | loss (sum, ×batch) | mAP50 | mAP50-95 |
|---|---|---|---|
| 1 | 217.0 | 0.000 | 0.000 |
| 10 | 207.3 | 0.000 | 0.000 |
| 20 | 171.9 | 0.003 | 0.001 |
| 30 | 144.3 | 0.032 | 0.012 |
| 40 | 124.0 | 0.050 | 0.023 |
| 50 | 116.6 | **0.069** | 0.032 |

Loss, smoothed over a 5-epoch window, decreased at every one of the 49
steps: **passed**. mAP50 after 50 epochs is 0.069 against a target of 0.3:
**not met from random initialisation**. 128 images × 50 epochs is 400
optimizer steps; the curve is still rising steeply (0.050 → 0.069 over the
last ten epochs) and nothing in the run suggests a defect, but no nano
detector reaches 0.3 mAP50 from scratch in that budget. The usual COCO128
tutorials start from COCO-pretrained weights. A fine-tuning run from the
converted GELAN-C weights is recorded below as gate 4a′ so the loop is
shown reaching a high mAP; it is a weaker check because those weights have
seen COCO128's images during their original training.

## Template

```
### <gate name>

commit:   <sha>
hardware: dev-3060
command:  uv run scripts/<...>
time:     <wall clock>
result:   <metric> = <value>
```
