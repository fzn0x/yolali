"""Exponential moving average of model weights."""

from __future__ import annotations

import copy
import math
from typing import Any

import torch
from torch import nn


class ModelEMA:
    """Keep ``ema = decay * ema + (1 - decay) * model`` after every optimizer step.

    The effective decay ramps up as ``decay * (1 - exp(-updates / tau))`` so
    the average starts close to the live weights instead of the random init.
    Buffers (BatchNorm statistics) are copied, not averaged.

    Args:
        model: the live model; a deep copy becomes the EMA.
        decay: steady-state decay.
        tau: ramp constant in updates.
    """

    def __init__(self, model: nn.Module, decay: float = 0.9999, tau: float = 2000.0) -> None:
        self.module = copy.deepcopy(model).eval()
        for p in self.module.parameters():
            p.requires_grad_(False)
        self.decay = decay
        self.tau = tau
        self.updates = 0

    def current_decay(self) -> float:
        """Decay used for the next update."""
        return self.decay * (1 - math.exp(-self.updates / self.tau))

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        """Blend ``model`` into the average."""
        self.updates += 1
        d = self.current_decay()
        live = model.state_dict()
        for name, value in self.module.state_dict().items():
            if value.dtype.is_floating_point:
                value.mul_(d).add_(live[name].detach().to(value.dtype), alpha=1 - d)
            else:
                value.copy_(live[name])

    def state_dict(self) -> dict[str, Any]:
        """Weights plus the update counter, for checkpoints."""
        return {"module": self.module.state_dict(), "updates": self.updates}

    def load_state_dict(self, state: dict[str, Any]) -> None:
        """Inverse of :meth:`state_dict`."""
        self.module.load_state_dict(state["module"])
        self.updates = int(state["updates"])
