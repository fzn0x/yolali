# Model, backbones, heads

`YolaliModel` is three attributes and a `forward`:

```python
class YolaliModel(nn.Module):
    def forward(self, x: Tensor) -> RawPrediction | DualRawPrediction:
        return self.head(self.neck(self.backbone(x)))
```

`build_model(ModelConfig)` assembles it from a flat YAML:

```yaml
name: gelan-n
nc: 80
backbone: { type: gelan, width: 0.25, depth: 0.33 }
neck:     { type: gelan_pan, width: 0.25, depth: 0.33 }
head:     { type: dual, reg_max: 0 }
```

Builders live in plain dictionaries (`BACKBONES`, `NECKS`, `HEADS`); adding
a backbone is adding a key.

## Backbones

Every backbone is a `Backbone` (an `nn.Module` with `out_channels`,
`strides` and `forward(x) -> [P3, P4, P5]`).

**GELANBackbone** is the GELAN-C topology from YOLOv9 written as named
attributes, so `print(model.backbone)` is the network diagram:

```
stem1  Conv(3, 64, 3, s2)         /2
stem2  Conv(64, 128, 3, s2)       /4
stage2 RepCSPELAN(128, 256, 128, 64)
down3  ADown(256, 256)            /8
stage3 RepCSPELAN(256, 512, 256, 128)   -> P3
down4  ADown(512, 512)            /16
stage4 RepCSPELAN(512, 512, 512, 256)   -> P4
down5  ADown(512, 512)            /32
stage5 RepCSPELAN(512, 512, 512, 256)   -> P5
```

`width` scales every channel count (rounded to a multiple of 8) and
`depth` scales the bottleneck repeats. The blocks (`Conv`, `RepConv`,
`RepCSP`, `RepCSPELAN`, `ADown`, `SPPELAN`) are reimplemented from the
paper and cite their reference names; their parameter layouts match the
official checkpoints one-to-one, which is what the converter relies on.
`fuse()` folds BatchNorm and the RepConv branches for inference, with a
test that outputs do not change.

**ViTBackbone** loads a DINOv2 or DINOv3 small/base model through `timm`
at patch size 16 (DINOv2's patch-14 embedding is resampled) and feeds the
single 1/16 token grid to a `SpatialTuningAdapter`, which builds P3 with a
2× transposed convolution, P4 with a 1×1 projection and P5 with a stride-2
convolution. `freeze: true` stops gradients into the transformer and keeps
it in eval mode.

## Neck

`GELANPAN` is the "head" half of the GELAN-C config: SPP-ELAN on P5, two
top-down merges, two bottom-up merges, all `RepCSPELAN`. It takes
`in_channels` from the backbone, so it sits behind either backbone.

## Heads

**DetectHead** has a box branch and a class branch per level, each two
3×3 convs and a 1×1, laid out like YOLOv9's `DDetect` (the box branch uses
four groups when regressing DFL bins). It returns `RawPrediction(box_raw,
cls_logits, anchors, strides)` with the levels flattened in stride order.
Decoding is not done here; see [ops](ops.md).

**DualDetectHead** holds two `DetectHead`s. While training it returns
`DualRawPrediction(o2m, o2o)`; in `eval()` it returns the one-to-one
branch alone. The one-to-one branch consumes **detached** neck features
by default (deviation D3): the richer one-to-many supervision shapes the
trunk, and the one-to-one head learns to pick its single anchor on top.

## Converting official weights

`yolali.zoo` reads the YOLOv9 authors' checkpoints without importing their
code: a stub-class unpickler reconstructs the pickled module tree, and a
prefix table (`model.3.` → `backbone.down3.`, …) renames every key. Two
tests cover it: a CI-safe one that maps the reference key/shape table onto
our `state_dict`, and one that runs the converted model against reference
forward-pass dumps when the checkpoints are present. The reference's
BatchNorm uses `eps=1e-3, momentum=0.03`; so does ours, and the
[design](../design.md) records why.

Source: `src/yolali/nn/`, `src/yolali/zoo/`. API: [yolali.nn](../api/nn.md), [yolali.zoo](../api/zoo.md).
