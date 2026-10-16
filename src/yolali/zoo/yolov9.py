"""Convert official GELAN-C / YOLOv9-C checkpoints into a yolali state dict.

The reference model is a flat ``nn.Sequential`` indexed by layer number
(``model.0`` ... ``model.22``); yolali names each block. The table below is
the whole conversion: every reference prefix on the left becomes the yolali
prefix on the right, and the remainder of each key is identical because the
blocks share internal attribute names (``cv1``, ``m.0.cv2.conv``, ...).

Weights converted this way remain under the YOLOv9 authors' GPL-3.0 license.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

from torch import Tensor

from yolali.zoo.unpickle import load_foreign_state_dict

# (reference layer prefix, yolali attribute prefix) for the GELAN-C graph.
GELAN_C_PREFIXES: tuple[tuple[str, str], ...] = (
    ("model.0.", "backbone.stem1."),
    ("model.1.", "backbone.stem2."),
    ("model.2.", "backbone.stage2."),
    ("model.3.", "backbone.down3."),
    ("model.4.", "backbone.stage3."),
    ("model.5.", "backbone.down4."),
    ("model.6.", "backbone.stage4."),
    ("model.7.", "backbone.down5."),
    ("model.8.", "backbone.stage5."),
    ("model.9.", "neck.spp."),
    ("model.12.", "neck.up_p4."),
    ("model.15.", "neck.up_p3."),
    ("model.16.", "neck.down_p4."),
    ("model.18.", "neck.out_p4."),
    ("model.19.", "neck.down_p5."),
    ("model.21.", "neck.out_p5."),
    ("model.22.cv2.", "head.box."),
    ("model.22.cv3.", "head.cls."),
)

# Reference tensors with no yolali counterpart: the DFL "conv" is a frozen
# arange that yolali computes in ops.decode instead of storing.
GELAN_C_IGNORED: tuple[str, ...] = ("model.22.dfl.conv.weight",)


def convert_gelan_c_state_dict(reference: Mapping[str, Tensor]) -> dict[str, Tensor]:
    """Rename reference keys to yolali keys; raise on anything unexpected."""
    out: dict[str, Tensor] = {}
    unmatched: list[str] = []
    for key, value in reference.items():
        if key in GELAN_C_IGNORED:
            continue
        for ref_prefix, our_prefix in GELAN_C_PREFIXES:
            if key.startswith(ref_prefix):
                out[our_prefix + key[len(ref_prefix) :]] = value
                break
        else:
            unmatched.append(key)
    if unmatched:
        raise ValueError(f"{len(unmatched)} reference keys have no mapping, e.g. {unmatched[:5]}")
    return out


def load_yolov9_gelan_c(path: str | Path) -> dict[str, Tensor]:
    """Read an official ``gelan-c.pt`` / ``yolov9-c-converted.pt`` as a yolali state dict."""
    return convert_gelan_c_state_dict(load_foreign_state_dict(path))
