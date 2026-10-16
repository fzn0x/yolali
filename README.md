<!-- banner: assets/banner.png (added in phase 8) -->

# yolali

**A YOLO you can actually read.**

A clean-room PyTorch object detection library targeting the 2026 state of the
art: NMS-free end-to-end inference by default, optional DFL, pluggable GELAN
and DINOv2/DINOv3 backbones, and a loss that is decoupled from the model.
Every component is a plain `nn.Module` or a typed dataclass, fully annotated,
tested against hand-computed values, and small enough to understand in one
sitting.

> **Status: pre-alpha.** The design document and phased plan are in
> [`docs/design.md`](docs/design.md) and [`docs/plan.md`](docs/plan.md).
> No library code is committed until the plan is approved. Nothing in this
> README describes a measured number until `BENCHMARKS.md` records one.

## Why another YOLO

The official reference implementations are research code. They work, but
reading them to learn *how* a modern detector is assembled is a multi-day
effort. yolali is a library, not a framework: no global state, no
monkey-patching, no `sys.argv` parsing anywhere under `src/`. You import
the pieces you want and swap the ones you don't.

## 30-second quickstart

The API below is the contract the implementation will be held to. It will
become runnable at the end of phase 7 (see `docs/plan.md`).

```python
from yolali import Detector, load_model

model = load_model("configs/models/gelan-n.yaml", weights="runs/voc/best.pt")
det = Detector(model, device="cuda")          # "cpu" and "mps" also work

results = det.predict(["dog.jpg", "street.jpg"], conf=0.25)
for r in results:                              # one Detections per image
    print(r.xyxy, r.conf, r.cls)              # (N, 4) float, (N,) float, (N,) int

model.export(format="onnx", path="gelan-n.onnx")   # no custom ops, no NMS
```

```bash
uv run scripts/detect.py --weights runs/voc/best.pt --source dog.jpg
```

## What "current" means here

| Requirement | Approach |
|---|---|
| NMS-free inference | One-to-many + one-to-one dual label assignment during training (YOLOv10 style); the one-to-one head serves inference with no NMS. A classic head + NMS path stays behind a config flag. |
| DFL optional | Both DFL (YOLOv8/v9 style) and direct `ltrb` regression (YOLO26 style). The direct head exports to ONNX with standard ops only. |
| Pluggable backbones | GELAN CNN (YOLOv9 lineage) and a DINOv2/DINOv3 small ViT with a Spatial Tuning Adapter (DEIMv2 style). Both emit `list[Tensor]` at strides 8, 16, 32. |
| Decoupled loss | `LossConfig` dataclass: task-aligned assigner, CIoU/GIoU, BCE or varifocal, optional DFL, one-to-one loss. |
| Verifiably correct | Weight converter + forward-pass equivalence test against official GELAN-C checkpoints; COCO val2017 mAP reproduced through our own evaluator before any training. |

## Pretrained weights and licenses

yolali's code is **Apache-2.0**. The GELAN weights our converter can read are
published by WongKinYiu in the [YOLOv9 repository](https://github.com/WongKinYiu/yolov9)
and carry **its GPL-3.0 license**. They are never bundled here; you download
them yourself and the converter re-keys them into our module layout.

## Acknowledgements

The primary architectural inspiration for this library is the work of
**Chien-Yao Wang (WongKinYiu)** and collaborators on
[YOLOv9](https://arxiv.org/abs/2402.13616): the GELAN backbone and neck,
and Programmable Gradient Information. Module docstrings that reimplement
his ideas cite the paper and repository.

Ideas adopted from other authors, with thanks:

- **YOLOv10** (Wang et al., [arXiv:2405.14458](https://arxiv.org/abs/2405.14458)):
  consistent dual label assignment for NMS-free detection.
- **YOLO26** (Ultralytics release notes): removing DFL in favour of direct
  box regression for simpler, op-free export.
- **D-FINE** (Peng et al., [arXiv:2410.13842](https://arxiv.org/abs/2410.13842))
  and **DEIMv2** (Huang et al., [arXiv:2509.20787](https://arxiv.org/abs/2509.20787)):
  the Spatial Tuning Adapter that turns single-scale ViT features into a pyramid.
- **RF-DETR** (Roboflow, [roboflow/rf-detr](https://github.com/roboflow/rf-detr)):
  a clean reference for a DINOv2-backbone real-time detector.

Everything here is written from scratch. No code was taken from AGPL or GPL
repositories; see `NOTICE`.

## License

Apache-2.0. See [`LICENSE`](LICENSE) and [`NOTICE`](NOTICE).
