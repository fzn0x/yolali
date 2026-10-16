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

| Label | CPU | GPU | RAM | OS |
|---|---|---|---|---|
| `dev-3060` | (filled in at first measurement) | NVIDIA GeForce RTX 3060 12 GB, driver 576.80 | | Windows 11 Pro |

## Template

```
### <gate name>

commit:   <sha>
hardware: dev-3060
command:  uv run scripts/<...>
time:     <wall clock>
result:   <metric> = <value>
```
