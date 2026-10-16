"""Optimizer construction and the warmup + cosine learning-rate schedule.

Three parameter groups follow the YOLO lineage: BatchNorm weights (no
decay), other weights (decay), biases (no decay, higher warmup start).
"""

from __future__ import annotations

import math

import torch
from torch import nn

from yolali.train.config import TrainConfig


def param_groups(model: nn.Module, weight_decay: float) -> list[dict[str, object]]:
    """Split parameters into ``[decayed weights, undecayed norm weights, biases]``."""
    decay: list[nn.Parameter] = []
    no_decay: list[nn.Parameter] = []
    biases: list[nn.Parameter] = []
    norm_types = (nn.BatchNorm2d, nn.GroupNorm, nn.LayerNorm)
    for module in model.modules():
        for name, p in module.named_parameters(recurse=False):
            if not p.requires_grad:
                continue
            if name.endswith("bias"):
                biases.append(p)
            elif isinstance(module, norm_types) or p.ndim <= 1:
                no_decay.append(p)
            else:
                decay.append(p)
    return [
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
        {"params": biases, "weight_decay": 0.0},
    ]


def build_optimizer(model: nn.Module, cfg: TrainConfig) -> torch.optim.Optimizer:
    """SGD (nesterov) or AdamW over :func:`param_groups`."""
    groups = param_groups(model, cfg.weight_decay)
    if cfg.optimizer == "adamw":
        return torch.optim.AdamW(groups, lr=cfg.lr, betas=(cfg.momentum, 0.999))
    return torch.optim.SGD(groups, lr=cfg.lr, momentum=cfg.momentum, nesterov=True)


class LRSchedule:
    """Per-iteration learning rate and momentum.

    During warmup every group's lr rises linearly to ``lr * factor(epoch)``;
    biases start at ``warmup_bias_lr`` instead of 0 and momentum rises from
    ``warmup_momentum``. After warmup only the epoch factor applies.

    Args:
        cfg: training config.
        iters_per_epoch: batches per epoch (before accumulation).
    """

    def __init__(self, cfg: TrainConfig, iters_per_epoch: int) -> None:
        self.cfg = cfg
        self.iters_per_epoch = iters_per_epoch
        self.warmup_iters = max(round(cfg.warmup_epochs * iters_per_epoch), cfg.warmup_min_iters)

    def epoch_factor(self, epoch: float) -> float:
        """Multiplier on ``lr`` at a (fractional) epoch, in ``[final_lr_factor, 1]``."""
        f = self.cfg.final_lr_factor
        progress = min(epoch / max(self.cfg.epochs, 1), 1.0)
        if self.cfg.cosine:
            return ((1 - math.cos(math.pi * progress)) / 2) * (f - 1) + 1
        return (1 - progress) * (1 - f) + f

    def apply(self, optimizer: torch.optim.Optimizer, iteration: int) -> float:
        """Set lr/momentum on every group for global ``iteration``; returns the weight lr."""
        epoch = iteration / max(self.iters_per_epoch, 1)
        lr = self.cfg.lr * self.epoch_factor(math.floor(epoch))
        for i, group in enumerate(optimizer.param_groups):
            if iteration < self.warmup_iters:
                t = iteration / self.warmup_iters
                start = self.cfg.warmup_bias_lr if i == 2 else 0.0
                group["lr"] = start + (lr - start) * t
                if "momentum" in group:
                    group["momentum"] = (
                        self.cfg.warmup_momentum
                        + (self.cfg.momentum - self.cfg.warmup_momentum) * t
                    )
            else:
                group["lr"] = lr
                if "momentum" in group:
                    group["momentum"] = self.cfg.momentum
        return float(optimizer.param_groups[0]["lr"])
