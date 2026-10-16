"""Convert an official GELAN-C / YOLOv9-C checkpoint to a yolali state dict.

Example:
    uv run scripts/convert_yolov9.py weights/gelan-c.pt weights/gelan-c.yolali.pt

The converted file stays under the YOLOv9 authors' GPL-3.0 license.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from yolali.nn.config import ModelConfig
from yolali.nn.model import build_model
from yolali.zoo import load_yolov9_gelan_c

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("source", type=Path, help="official gelan-c.pt or yolov9-c-converted.pt")
    parser.add_argument("target", type=Path, help="output .pt holding a yolali state_dict")
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs" / "models" / "gelan-c.yaml",
        help="model config used to validate the converted keys",
    )
    args = parser.parse_args()

    state = load_yolov9_gelan_c(args.source)
    model = build_model(ModelConfig.from_yaml(args.config))
    model.load_state_dict(state, strict=True)
    args.target.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, args.target)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"wrote {args.target} ({len(state)} tensors, {n_params:,} parameters)")


if __name__ == "__main__":
    main()
