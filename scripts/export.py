"""Export a model to ONNX or TorchScript.

    uv run scripts/export.py --config configs/models/gelan-n.yaml --weights runs/voc/best.pt \\
        --format onnx --out gelan-n.onnx
"""

from __future__ import annotations

import argparse
from pathlib import Path

from yolali.nn.model import load_model


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--format", choices=["onnx", "torchscript"], default="onnx")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--img-size", type=int, default=640)
    parser.add_argument("--max-det", type=int, default=300)
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument("--no-fuse", action="store_true")
    args = parser.parse_args()

    model = load_model(args.config, args.weights)
    if not args.no_fuse:
        model.fuse()
    if args.format == "onnx":
        out = model.export(
            "onnx", args.out, img_size=args.img_size, max_det=args.max_det, opset=args.opset
        )
    else:
        out = model.export("torchscript", args.out, img_size=args.img_size, max_det=args.max_det)
    kind = (
        "NMS-free detections (B, max_det, 6)"
        if model.nms_free
        else "raw boxes + scores (B, A, 4 + nc)"
    )
    print(f"wrote {out} ({out.stat().st_size / 1e6:.1f} MB), output = {kind}")


if __name__ == "__main__":
    main()
