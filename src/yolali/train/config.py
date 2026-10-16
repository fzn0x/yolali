"""Training hyper-parameters as one frozen dataclass."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal


@dataclass(frozen=True)
class TrainConfig:
    """Everything the :class:`~yolali.train.Trainer` needs; each feature has a switch.

    Attributes:
        epochs: total epochs (resume continues to this number).
        img_size: training resolution; multi-scale varies around it.
        optimizer: ``"sgd"`` (YOLO lineage default) or ``"adamw"``.
        lr: peak learning rate.
        final_lr_factor: learning rate at the last epoch as a fraction of ``lr``.
        momentum: SGD momentum / AdamW beta1.
        weight_decay: applied to conv/linear weights only (not BN, not biases).
        warmup_epochs: linear warmup length; at least ``warmup_min_iters`` steps.
        warmup_min_iters: lower bound on warmup iterations.
        warmup_bias_lr: biases start here and decay to ``lr`` during warmup.
        warmup_momentum: momentum at the start of warmup.
        cosine: cosine decay after warmup; ``False`` means linear.
        amp: mixed precision (bf16 where supported, else fp16 with a scaler).
        ema: keep an exponential moving average of weights for eval and export.
        ema_decay: EMA decay at steady state.
        ema_tau: EMA ramp-up constant in updates.
        close_mosaic: switch ``train_loader.dataset.transform`` to
            ``close_mosaic_transform`` for the last N epochs (``0`` disables).
        accumulate: optimizer step every N batches.
        multi_scale: ``(min, max)`` size range (multiples of 32) sampled per
            batch; ``None`` disables.
        compile: wrap the model in ``torch.compile``.
        grad_clip: max gradient norm; ``0`` disables clipping.
        val_every: evaluate every N epochs (``0`` disables; the last epoch
            always evaluates when an evaluator is present).
        seed: base seed for the model init, loaders and multi-scale draws.
        out_dir: where ``last.pt``, ``best.pt`` and logs go.
        log_every: log training scalars every N iterations.
    """

    epochs: int
    img_size: int = 640
    optimizer: Literal["sgd", "adamw"] = "sgd"
    lr: float = 0.01
    final_lr_factor: float = 0.01
    momentum: float = 0.937
    weight_decay: float = 5e-4
    warmup_epochs: float = 3.0
    warmup_min_iters: int = 100
    warmup_bias_lr: float = 0.1
    warmup_momentum: float = 0.8
    cosine: bool = True
    amp: bool = True
    ema: bool = True
    ema_decay: float = 0.9999
    ema_tau: float = 2000.0
    close_mosaic: int = 10
    accumulate: int = 1
    multi_scale: tuple[int, int] | None = None
    compile: bool = False
    grad_clip: float = 10.0
    val_every: int = 1
    seed: int = 0
    out_dir: Path = Path("runs/exp")
    log_every: int = 10

    def __post_init__(self) -> None:
        if self.epochs < 1:
            raise ValueError("epochs must be >= 1")
        if self.accumulate < 1:
            raise ValueError("accumulate must be >= 1")
        if self.multi_scale is not None:
            lo, hi = self.multi_scale
            if lo > hi or lo % 32 or hi % 32:
                raise ValueError("multi_scale must be (min, max) multiples of 32 with min <= max")

    def as_dict(self) -> dict[str, Any]:
        """JSON-friendly mapping."""
        d = asdict(self)
        d["out_dir"] = str(self.out_dir)
        return d
