import math

import pytest
import torch
from torch import nn

from yolali.train.config import TrainConfig
from yolali.train.schedule import LRSchedule, build_optimizer, param_groups


def tiny_model() -> nn.Module:
    return nn.Sequential(
        nn.Conv2d(3, 4, 3, bias=True), nn.BatchNorm2d(4), nn.Conv2d(4, 2, 1, bias=False)
    )


def test_param_groups_split_decay():
    groups = param_groups(tiny_model(), 0.1)
    decayed, no_decay, biases = (sum(p.numel() for p in g["params"]) for g in groups)  # type: ignore[union-attr]
    assert decayed == 4 * 3 * 9 + 2 * 4  # two conv weights
    assert no_decay == 4  # BN weight
    assert biases == 4 + 4  # conv bias + BN bias
    assert groups[0]["weight_decay"] == 0.1 and groups[1]["weight_decay"] == 0.0


def test_epoch_factor_cosine_endpoints():
    s = LRSchedule(TrainConfig(epochs=10, final_lr_factor=0.01), iters_per_epoch=5)
    assert s.epoch_factor(0) == pytest.approx(1.0)
    assert s.epoch_factor(10) == pytest.approx(0.01)
    assert s.epoch_factor(5) == pytest.approx((1 + 0.01) / 2)


def test_epoch_factor_linear():
    s = LRSchedule(TrainConfig(epochs=10, final_lr_factor=0.0, cosine=False), 1)
    assert s.epoch_factor(2.5) == pytest.approx(0.75)


def test_warmup_ramps_and_biases_start_high():
    cfg = TrainConfig(
        epochs=4,
        lr=0.1,
        warmup_epochs=1,
        warmup_min_iters=4,
        warmup_bias_lr=0.5,
        warmup_momentum=0.8,
    )
    opt = build_optimizer(tiny_model(), cfg)
    s = LRSchedule(cfg, iters_per_epoch=4)
    assert s.warmup_iters == 4
    s.apply(opt, 0)
    assert opt.param_groups[0]["lr"] == 0.0
    assert opt.param_groups[2]["lr"] == pytest.approx(0.5)
    assert opt.param_groups[0]["momentum"] == pytest.approx(0.8)
    s.apply(opt, 2)  # halfway through warmup, still epoch 0 -> target lr 0.1
    assert opt.param_groups[0]["lr"] == pytest.approx(0.05)
    assert opt.param_groups[2]["lr"] == pytest.approx(0.5 + (0.1 - 0.5) * 0.5)
    lr = s.apply(opt, 4)  # warmup over, epoch 1
    assert lr == pytest.approx(0.1 * s.epoch_factor(1))
    assert opt.param_groups[0]["momentum"] == pytest.approx(0.937)


def test_adamw_option():
    cfg = TrainConfig(epochs=1, optimizer="adamw")
    assert isinstance(build_optimizer(tiny_model(), cfg), torch.optim.AdamW)


def test_config_validation():
    with pytest.raises(ValueError):
        TrainConfig(epochs=0)
    with pytest.raises(ValueError):
        TrainConfig(epochs=1, multi_scale=(100, 200))
    assert math.isclose(TrainConfig(epochs=1).as_dict()["lr"], 0.01)
