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

## Template

```
### <gate name>

commit:   <sha>
hardware: dev-3060
command:  uv run scripts/<...>
time:     <wall clock>
result:   <metric> = <value>
```
