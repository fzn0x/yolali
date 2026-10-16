"""The training loop.

One class, one ``fit``. Every optional feature is a single ``if cfg.flag``
at the place where it acts, so the loop can be read top to bottom:

    for epoch:
        maybe close mosaic
        for batch:
            maybe multi-scale
            forward (maybe autocast) -> loss
            backward (maybe scaled)
            every `accumulate` batches: maybe clip, step, maybe EMA
        maybe validate; save last / best
"""

from __future__ import annotations

import logging
import math
import os
import time
from collections.abc import Callable, Iterable, Sized
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F  # noqa: N812 - torch convention
from torch import Tensor, nn
from torch.amp.grad_scaler import GradScaler
from torch.nn.parallel import DistributedDataParallel

from yolali.data.collate import Batch
from yolali.data.transforms.base import Transform
from yolali.eval.evaluator import Evaluator
from yolali.eval.map import DetectionMetrics
from yolali.train.checkpoint import Checkpoint, RNGState, load_checkpoint, save_checkpoint
from yolali.train.config import TrainConfig
from yolali.train.ema import ModelEMA
from yolali.train.loggers.base import Logger, MultiLogger
from yolali.train.schedule import LRSchedule, build_optimizer
from yolali.types import Targets

log = logging.getLogger(__name__)


class LossLike(Protocol):
    """What a criterion's output must provide."""

    @property
    def total(self) -> Tensor:
        """Scalar to backpropagate."""
        ...

    def as_dict(self) -> dict[str, float]:
        """Detached terms for logging."""
        ...


# ``criterion(model_output, targets) -> LossLike``; the output type depends on the head.
Criterion = Callable[[Any, Targets], LossLike]


def fitness(metrics: DetectionMetrics) -> float:
    """The number ``best.pt`` tracks: ``0.1 * mAP50 + 0.9 * mAP50-95``."""
    return 0.1 * max(metrics.map50, 0.0) + 0.9 * max(metrics.map50_95, 0.0)


def resolve_device(device: str | torch.device) -> torch.device:
    """``"auto"`` -> cuda, mps or cpu, whichever is available first."""
    if device != "auto":
        return torch.device(device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def scale_batch(batch: Batch, size: int) -> Batch:
    """Resize images (and boxes) so the long side is ``size``; used by multi-scale."""
    _, _, h, w = batch.images.shape
    factor = size / max(h, w)
    if factor == 1.0:
        return batch
    new_h = math.ceil(h * factor / 32) * 32
    new_w = math.ceil(w * factor / 32) * 32
    images = F.interpolate(batch.images, size=(new_h, new_w), mode="bilinear", align_corners=False)
    sx, sy = new_w / w, new_h / h
    boxes = batch.targets.boxes * torch.tensor([sx, sy, sx, sy], device=batch.targets.boxes.device)
    targets = Targets(boxes, batch.targets.classes, batch.targets.mask)
    return Batch(images, targets, batch.paths, batch.orig_hw, batch.scales, batch.pads)


class Trainer:
    """Train ``model`` with ``criterion`` on ``train_loader``.

    Args:
        model: the detector (raw ``nn.Module``; EMA and DDP wrapping happen here).
        criterion: loss taking the model output and :class:`Targets`.
        train_loader: yields :class:`Batch`; its ``dataset`` may expose
            ``set_epoch`` and ``transform`` (used by close-mosaic).
        cfg: hyper-parameters.
        evaluator: optional; when given, validation runs every ``cfg.val_every``
            epochs on ``val_loader`` and ``best.pt`` tracks :func:`fitness`.
        val_loader: batches for the evaluator.
        loggers: any number of :class:`Logger` implementations.
        close_mosaic_transform: pipeline to swap in for the last
            ``cfg.close_mosaic`` epochs.
        device: ``"auto"`` or an explicit device.

    DDP: launch with ``torchrun``; the trainer reads ``RANK``/``WORLD_SIZE``,
    wraps the model and only rank 0 logs, validates and saves. The caller
    builds the loader with a ``DistributedSampler``.
    """

    def __init__(
        self,
        model: nn.Module,
        criterion: Criterion,
        train_loader: Iterable[Batch],
        cfg: TrainConfig,
        *,
        evaluator: Evaluator | None = None,
        val_loader: Iterable[Batch] | None = None,
        loggers: Iterable[Logger] = (),
        close_mosaic_transform: Transform | None = None,
        device: str | torch.device = "auto",
    ) -> None:
        self.cfg = cfg
        self.criterion = criterion
        self.train_loader = train_loader
        self.evaluator = evaluator
        self.val_loader = val_loader
        self.close_mosaic_transform = close_mosaic_transform
        self.logger = MultiLogger(list(loggers))

        self.world_size = int(os.environ.get("WORLD_SIZE", "1"))
        self.rank = int(os.environ.get("RANK", "0"))
        self.distributed = self.world_size > 1
        if self.distributed and not dist.is_initialized():
            dist.init_process_group("nccl" if torch.cuda.is_available() else "gloo")
            if torch.cuda.is_available():
                torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", "0")))
                device = torch.device("cuda", int(os.environ.get("LOCAL_RANK", "0")))
        self.device = resolve_device(device)

        self.model = model.to(self.device)
        self.ema = ModelEMA(self.model, cfg.ema_decay, cfg.ema_tau) if cfg.ema else None
        self.optimizer = build_optimizer(self.model, cfg)
        iters = len(train_loader) if isinstance(train_loader, Sized) else 1
        self.schedule = LRSchedule(cfg, iters)
        self.use_bf16 = self.device.type == "cuda" and torch.cuda.is_bf16_supported()
        self.amp_dtype = torch.bfloat16 if self.use_bf16 else torch.float16
        self.scaler = GradScaler(
            self.device.type, enabled=cfg.amp and self.device.type == "cuda" and not self.use_bf16
        )
        self.start_epoch = 0
        self.best_fitness = -1.0
        self.iteration = 0
        self._compiled: Callable[[Tensor], Any] | None = None
        self._ddp: DistributedDataParallel | None = None
        if self.distributed:
            self._ddp = DistributedDataParallel(
                self.model,
                device_ids=[self.device.index] if self.device.type == "cuda" else None,
            )

    # ------------------------------------------------------------------ helpers
    @property
    def is_main(self) -> bool:
        """Rank 0 (or not distributed)."""
        return self.rank == 0

    def forward_model(self) -> Callable[[Tensor], Any]:
        """The callable to run in the loop: DDP wrapper, compiled graph, or the model."""
        if self._ddp is not None:
            return self._ddp
        if self.cfg.compile:
            if self._compiled is None:
                self._compiled = torch.compile(self.model)  # pyright: ignore[reportUnknownMemberType] - returns a callable
            return self._compiled
        return self.model

    def eval_model(self) -> nn.Module:
        """EMA weights when enabled, else the live model."""
        return self.ema.module if self.ema is not None else self.model

    def _dataset_attr(self, name: str) -> Any:  # noqa: ANN401 - duck-typed loader
        return getattr(getattr(self.train_loader, "dataset", None), name, None)

    def _set_epoch(self, epoch: int) -> None:
        for owner in (
            self.train_loader,
            getattr(self.train_loader, "sampler", None),
            getattr(self.train_loader, "batch_sampler", None),
            getattr(self.train_loader, "dataset", None),
        ):
            set_epoch = getattr(owner, "set_epoch", None)
            if callable(set_epoch):
                set_epoch(epoch)

    # ------------------------------------------------------------------ checkpoints
    def checkpoint(self, epoch: int) -> Checkpoint:
        """Snapshot after ``epoch`` completed."""
        model_cfg = getattr(self.model, "cfg", None)
        return Checkpoint(
            model={k: v.detach().cpu() for k, v in self.model.state_dict().items()},
            epoch=epoch,
            optimizer=self.optimizer.state_dict(),
            ema=self.ema.state_dict() if self.ema is not None else None,
            scaler=self.scaler.state_dict() if self.scaler.is_enabled() else None,
            rng=RNGState.capture(),
            best_fitness=self.best_fitness,
            config=self.cfg.as_dict(),
            model_config=getattr(model_cfg, "__dict__", {}),
        )

    def resume(self, path: str | Path) -> None:
        """Load ``last.pt`` and continue from the next epoch, RNG included."""
        ckpt = load_checkpoint(path)
        self.model.load_state_dict(ckpt.model)
        if ckpt.optimizer is not None:
            self.optimizer.load_state_dict(ckpt.optimizer)
        if self.ema is not None and ckpt.ema is not None:
            self.ema.load_state_dict(ckpt.ema)
        if ckpt.scaler is not None and self.scaler.is_enabled():
            self.scaler.load_state_dict(ckpt.scaler)
        if ckpt.rng is not None:
            ckpt.rng.restore()
        self.best_fitness = ckpt.best_fitness
        self.start_epoch = ckpt.epoch + 1
        self.iteration = self.start_epoch * self.schedule.iters_per_epoch
        log.info("resumed from %s at epoch %d", path, self.start_epoch)

    def save(self, name: str, epoch: int) -> Path:
        """Write ``<out_dir>/<name>`` and return its path."""
        path = Path(self.cfg.out_dir) / name
        save_checkpoint(self.checkpoint(epoch), path)
        return path

    # ------------------------------------------------------------------ loop
    def train_step(self, batch: Batch, rng: np.random.Generator) -> dict[str, float]:
        """One batch: forward, loss, backward; optimizer step every ``accumulate`` batches."""
        cfg = self.cfg
        if cfg.multi_scale is not None:
            lo, hi = cfg.multi_scale
            size = int(rng.integers(lo // 32, hi // 32 + 1)) * 32
            batch = scale_batch(batch, size)
        batch = batch.to(self.device, non_blocking=True)

        lr = self.schedule.apply(self.optimizer, self.iteration)
        with torch.autocast(
            self.device.type, dtype=self.amp_dtype, enabled=cfg.amp and self.device.type != "cpu"
        ):
            pred = self.forward_model()(batch.images)
        out = self.criterion(pred, batch.targets)
        self.scaler.scale(out.total / cfg.accumulate).backward()

        if (self.iteration + 1) % cfg.accumulate == 0:
            if cfg.grad_clip > 0:
                self.scaler.unscale_(self.optimizer)
                nn.utils.clip_grad_norm_(self.model.parameters(), cfg.grad_clip)
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.optimizer.zero_grad(set_to_none=True)
            if self.ema is not None:
                self.ema.update(self.model)
        self.iteration += 1
        return {**{f"train/{k}": v for k, v in out.as_dict().items()}, "train/lr": lr}

    def train_epoch(self, epoch: int) -> dict[str, float]:
        """Run every batch once; returns the mean of the logged scalars."""
        cfg = self.cfg
        self.model.train()
        self._set_epoch(epoch)
        if (
            cfg.close_mosaic
            and epoch == cfg.epochs - cfg.close_mosaic
            and self.close_mosaic_transform is not None
        ):
            dataset = getattr(self.train_loader, "dataset", None)
            if dataset is not None and hasattr(dataset, "transform"):
                dataset.transform = self.close_mosaic_transform
                log.info("epoch %d: mosaic closed", epoch)
        rng = np.random.default_rng([cfg.seed, epoch])
        sums: dict[str, float] = {}
        count = 0
        t0 = time.perf_counter()
        for i, batch in enumerate(self.train_loader):
            scalars = self.train_step(batch, rng)
            for k, v in scalars.items():
                sums[k] = sums.get(k, 0.0) + v
            count += 1
            if self.is_main and cfg.log_every and (i + 1) % cfg.log_every == 0:
                self.logger.log_scalars(self.iteration, scalars)
        means = {k: v / max(count, 1) for k, v in sums.items()}
        means["train/epoch_seconds"] = time.perf_counter() - t0
        return means

    def validate(self) -> DetectionMetrics | None:
        """Evaluate the EMA (or live) model on ``val_loader``."""
        if self.evaluator is None or self.val_loader is None:
            return None
        self.evaluator.model = self.eval_model()
        self.evaluator.device = self.device
        return self.evaluator.run(self.val_loader)

    def fit(self) -> dict[str, float]:
        """Train from ``start_epoch`` to ``cfg.epochs``; returns the last epoch's scalars."""
        cfg = self.cfg
        if self.is_main:
            self.logger.log_config(cfg.as_dict())
        last: dict[str, float] = {}
        for epoch in range(self.start_epoch, cfg.epochs):
            means = self.train_epoch(epoch)
            epoch_scalars = {f"epoch/{k.removeprefix('train/')}": v for k, v in means.items()}
            is_last = epoch == cfg.epochs - 1
            if (
                self.is_main
                and self.evaluator is not None
                and (is_last or (cfg.val_every and (epoch + 1) % cfg.val_every == 0))
            ):
                metrics = self.validate()
                if metrics is not None:
                    epoch_scalars.update({f"val/{k}": v for k, v in metrics.as_dict().items()})
                    fit = fitness(metrics)
                    epoch_scalars["val/fitness"] = fit
                    if fit > self.best_fitness:
                        self.best_fitness = fit
                        self.save("best.pt", epoch)
            if self.is_main:
                self.logger.log_scalars(epoch, epoch_scalars)
                self.save("last.pt", epoch)
                log.info(
                    "epoch %d/%d %s",
                    epoch + 1,
                    cfg.epochs,
                    " ".join(f"{k}={v:.4g}" for k, v in epoch_scalars.items()),
                )
            last = epoch_scalars
        if self.is_main:
            self.logger.close()
        if self.distributed:
            dist.destroy_process_group()
        return last
