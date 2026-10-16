# Losses

The loss is decoupled from the model: it consumes what the head emits
(`RawPrediction`) and the padded ground truth (`Targets`), and nothing
else. `LossConfig` is a frozen dataclass with every knob:

```python
@dataclass(frozen=True)
class LossConfig:
    box_weight: float = 7.5
    cls_weight: float = 0.5
    dfl_weight: float = 1.5          # ignored when reg_max == 0
    iou_kind: Literal["ciou", "giou"] = "ciou"
    cls_loss: Literal["bce", "vfl"] = "bce"
    reg_max: int = 16                # 0 => direct ltrb regression
    tal_topk: int = 10
    tal_alpha: float = 0.5
    tal_beta: float = 6.0
    o2o_weight: float = 1.0          # weight of the one-to-one branch in DualLoss
    vfl_alpha: float = 0.75
    vfl_gamma: float = 2.0
```

## DetectionLoss

`DetectionLoss(cfg, num_classes, assigner=None)(pred, targets)` returns a
`LossOutput(total, box, cls, dfl, num_fg)`:

- **cls**: BCE on the soft targets (or varifocal), summed, divided by the
  total target mass (clamped to at least 1).
- **box**: `1 − CIoU` on positive anchors, weighted by each anchor's soft
  target, same normalisation.
- **dfl** (only when `reg_max > 0`): cross-entropy against the two integer
  bins bracketing each side distance, weighted by `ceil(t) − t` and
  `t − floor(t)`, averaged over the four sides.
- **total** `= (7.5·box + 0.5·cls + 1.5·dfl) · batch_size`, the YOLOv8/v9
  convention that keeps the default weights and learning rate meaningful
  across batch sizes.

When no anchor is positive, `box` and `dfl` are `raw.sum() * 0` so the
graph stays connected and DDP never sees an unused parameter.

The closed-form tests in `tests/loss/test_detection.py` use a single
anchor at (4, 4) on a stride-8 map. With a direct-regression head whose
output decodes exactly to the ground truth, the box term is 0, the soft
target is 1, and `cls = BCE(0, 1) = ln 2`, so `total = 0.5·ln 2`.

## DualLoss

`DualLoss(cfg, num_classes)` holds two `DetectionLoss`es, one with the
`TaskAlignedAssigner` and one with the `OneToOneAssigner`, and returns
`o2m.total + o2o_weight · o2o.total` together with both breakdowns, so a
training log shows `o2m/box`, `o2o/cls`, and so on side by side.

Source: `src/yolali/loss/`. API: [yolali.loss](../api/loss.md).
