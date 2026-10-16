import math

import torch
from torch import nn

from yolali.train.ema import ModelEMA


def test_ema_decay_ramp_and_update():
    model = nn.Linear(2, 1, bias=False)
    with torch.no_grad():
        model.weight.fill_(1.0)
    ema = ModelEMA(model, decay=0.9, tau=1.0)
    with torch.no_grad():
        model.weight.fill_(0.0)
    ema.update(model)
    d = 0.9 * (1 - math.exp(-1))  # decay used by the first update
    weight = ema.module.state_dict()["weight"]
    assert torch.allclose(weight, torch.full((1, 2), d * 1.0 + (1 - d) * 0.0))
    assert not ema.module.weight.requires_grad


def test_ema_copies_buffers_and_roundtrips_state():
    model = nn.BatchNorm1d(3)
    ema = ModelEMA(model, decay=0.5, tau=1e-9)  # decay ~ 0.5 from the first update
    live = model.state_dict()
    live["running_mean"].fill_(2.0)
    live["num_batches_tracked"].fill_(7)
    ema.update(model)
    averaged = ema.module.state_dict()
    assert torch.allclose(averaged["running_mean"], torch.full((3,), 1.0))  # (0 + 2) / 2
    assert averaged["num_batches_tracked"].item() == 7  # copied, not averaged
    other = ModelEMA(nn.BatchNorm1d(3))
    other.load_state_dict(ema.state_dict())
    assert other.updates == 1
    assert torch.allclose(other.module.state_dict()["running_mean"], averaged["running_mean"])
