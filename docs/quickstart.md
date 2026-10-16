# Quickstart

Three things people do with yolali in the first ten minutes: run a converted
official model on an image, train a small model on a small dataset, and
export the result without NMS.

## Install

```bash
git clone https://github.com/fzn0x/yolali && cd yolali
uv sync --all-extras          # torch, torchvision, onnxruntime, pycocotools, timm, ...
uv run pytest                 # 160+ unit tests, CPU only, under a minute
```

yolali is not on PyPI yet. The package name is `yolali`; `uv add yolali`
will work once version 0.1.0 is cut.

## Detect with official GELAN-C weights

The YOLOv9 authors publish `gelan-c.pt` under GPL-3.0. yolali never bundles
it; the converter re-keys it into our module layout (and checks every
tensor shape against the config).

```bash
curl -L -o weights/gelan-c.pt \
  https://github.com/WongKinYiu/yolov9/releases/download/v0.1/gelan-c.pt
uv run scripts/convert_yolov9.py weights/gelan-c.pt weights/gelan-c.yolali.pt
uv run scripts/detect.py --config configs/models/gelan-c.yaml \
  --weights weights/gelan-c.yolali.pt --names configs/data/coco128.yaml \
  --source datasets/coco128/images/train2017/000000000009.jpg
```

The same thing from Python:

```python
from yolali import Detector, load_model

model = load_model("configs/models/gelan-c.yaml", weights="weights/gelan-c.yolali.pt")
detector = Detector(model, device="auto")         # cuda, mps or cpu

for result in detector.predict(["dog.jpg", "street.jpg"], conf=0.25):
    print(result.xyxy)      # (N, 4) float32, pixels of the original image
    print(result.conf)      # (N,)   float32
    print(result.cls)       # (N,)   int64
```

`Detections` is a frozen dataclass; nothing else comes back.

## Train on COCO128

```bash
uv run scripts/download_datasets.py coco128
uv run scripts/train.py --model configs/models/gelan-n.yaml \
  --data configs/data/coco128.yaml --epochs 50 --batch-size 16 --out runs/coco128
```

`gelan-n.yaml` is the NMS-free default (dual head, direct box regression).
`gelan-n-nms.yaml` is the classic DFL head evaluated with NMS, kept for
comparison. Everything the trainer does is a switch on
[`TrainConfig`][yolali.train.TrainConfig]; the script just fills it from
flags.

What a run writes:

| File | What |
|---|---|
| `runs/coco128/metrics.csv` | one row per logged step and per epoch, every loss term and every metric |
| `runs/coco128/last.pt` | model, EMA, optimizer, scaler and RNG state: `--resume` continues bitwise |
| `runs/coco128/best.pt` | highest `0.1·mAP50 + 0.9·mAP50-95` on the validation set |
| `runs/coco128/config.json` | the exact `TrainConfig` |

## Evaluate

```bash
uv run scripts/val.py --config configs/models/gelan-n.yaml --weights runs/coco128/best.pt \
  --images datasets/coco128/images/train2017 --no-nms
```

The evaluator is yolali's own COCO-protocol implementation; it agrees with
pycocotools to 1e-6 on every summary number (there is a test for that) and
reproduces the published GELAN-C result on val2017 (see
[Benchmarks](benchmarks.md)).

## Export

```bash
uv run scripts/export.py --config configs/models/gelan-n.yaml \
  --weights runs/coco128/best.pt --format onnx --out gelan-n.onnx
```

For the NMS-free head the ONNX graph's output is already the final
detections, `(batch, max_det, 6)` as `x1, y1, x2, y2, conf, cls` in
letterboxed pixels, using only standard ONNX ops (the test checks the
op domains and ONNX Runtime parity). Or in Python:

```python
model.export("onnx", "gelan-n.onnx")
model.export("torchscript", "gelan-n.torchscript")
```
