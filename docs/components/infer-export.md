# Inference and export

## Detector

```python
detector = Detector(model, device="auto", half=False, fuse=True, names=COCO_NAMES)
results = detector.predict(images, conf=0.25, iou=0.7, max_det=300, size=640)
```

`images` may be a path, a PIL image, an `(H, W, 3)` uint8 RGB array, a
`(3, H, W)` tensor, or a sequence of those. Every input is letterboxed to
a square, the batch runs once, the same `postprocess` as the evaluator
turns scores into candidates (NMS for a classic head, top-k for the dual
head), and the letterbox is undone so boxes come back in original pixels.

`Detections` is frozen: `xyxy (N, 4)`, `conf (N,)`, `cls (N,)`, optional
`names`, `orig_hw`. It iterates as `(box, conf, cls)` triples, has
`filter(min_conf)`, `label(i)`, `to(device)`. `draw_detections(image,
dets)` renders boxes for the CLI.

## ONNX

`model.export("onnx", path, img_size=640, max_det=300, opset=18)` wraps
the model in `ExportWrapper`, which includes decoding:

- NMS-free (dual head): output `(B, max_det, 6)` = `x1, y1, x2, y2, conf,
  cls` in letterboxed pixels, obtained with a `TopK` over per-anchor max
  scores. No NMS, no custom ops.
- Classic head: output `(B, A, 4 + nc)`, boxes then class probabilities;
  run your own NMS.

The exporter is the `torch.export`-based one (`dynamo=True`). The test
exports both head types, checks that every node is in the default ONNX
domain, runs ONNX Runtime, and asserts max abs diff `< 1e-4` against the
wrapper in PyTorch.

## TorchScript

`model.export("torchscript", path)` traces the same wrapper; the test
asserts parity with the eager wrapper.

## CLI

```bash
uv run scripts/detect.py --config configs/models/gelan-c.yaml \
  --weights weights/gelan-c.yolali.pt --names configs/data/coco128.yaml \
  --source some/dir --out runs/detect
uv run scripts/export.py --config configs/models/gelan-n.yaml \
  --weights runs/voc/best.pt --format onnx --out gelan-n.onnx
```

Source: `src/yolali/infer/`, `src/yolali/export/`. API: [yolali.infer](../api/infer.md), [yolali.export](../api/export.md).
