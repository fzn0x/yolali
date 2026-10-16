# Weights and licenses

yolali's **code is Apache-2.0**. Weights are a separate matter.

| Weights | Source | License | How yolali uses them |
|---|---|---|---|
| `gelan-c.pt`, `yolov9-c-converted.pt` | [WongKinYiu/yolov9 releases](https://github.com/WongKinYiu/yolov9/releases) | GPL-3.0 | downloaded by you, converted by `scripts/convert_yolov9.py`; never bundled |
| DINOv2 ViT-S/14, ViT-B/14 | `timm` (`vit_small_patch14_dinov2.lvd142m`, `vit_base_patch14_dinov2.lvd142m`) | Apache-2.0 | downloaded on first use by the ViT backbone |
| DINOv3 ViT-S/16 and larger | `timm` (`vit_small_patch16_dinov3.lvd1689m`, …) | Meta DINOv3 license | opt-in by model name in the YAML |
| Anything you train with yolali | you | yours | checkpoints are plain `state_dict`s plus training state |

A model converted from GPL-3.0 weights remains GPL-3.0, whatever code
loads it. Keep that in mind before shipping `gelan-c.yolali.pt` anywhere.

## What the converter does

1. Unpickles the official checkpoint with stub classes standing in for the
   reference repository's modules, so none of that GPL code is imported.
2. Walks the module tree and collects every parameter and buffer.
3. Renames keys with a prefix table (`model.0.` → `backbone.stem1.`, …)
   and drops the frozen DFL `arange` that yolali computes instead of
   storing.
4. Loads the result with `strict=True` into a model built from
   `configs/models/gelan-c.yaml`.

The forward pass of the converted model matches the reference to
`< 1e-4` (gate 2) and reproduces the published COCO mAP (gate 3); both
numbers are in [Benchmarks](benchmarks.md).

## Credits

The GELAN backbone, PAN neck, head layout and checkpoint format follow
**Chien-Yao Wang, I-Hau Yeh and Hong-Yuan Mark Liao, "YOLOv9: Learning
What You Want to Learn Using Programmable Gradient Information"**
([arXiv:2402.13616](https://arxiv.org/abs/2402.13616)). Dual assignment
follows YOLOv10 ([arXiv:2405.14458](https://arxiv.org/abs/2405.14458)).
The Spatial Tuning Adapter follows DEIMv2
([arXiv:2509.20787](https://arxiv.org/abs/2509.20787)); D-FINE
([arXiv:2410.13842](https://arxiv.org/abs/2410.13842)) and RF-DETR informed
the ViT path. YOLO26's release notes motivated the direct-regression head.
