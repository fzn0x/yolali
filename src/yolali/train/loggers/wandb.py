"""Weights & Biases logger (optional extra ``wandb``)."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any


class WandbLogger:
    """Log to a W&B run created on construction."""

    def __init__(self, project: str, name: str | None = None, **init_kwargs: Any) -> None:  # noqa: ANN401 - passthrough
        try:
            import wandb  # noqa: PLC0415 # pyright: ignore[reportMissingImports] - optional dependency
        except ImportError as exc:
            raise ImportError("WandbLogger needs `uv sync --extra wandb`") from exc
        self.wandb: Any = wandb
        self.run: Any = wandb.init(project=project, name=name, **init_kwargs)

    def log_scalars(self, step: int, values: Mapping[str, float]) -> None:
        """See :class:`~yolali.train.loggers.Logger`."""
        self.wandb.log(dict(values), step=step)

    def log_config(self, config: Mapping[str, Any]) -> None:
        """Update the run config."""
        self.run.config.update(dict(config), allow_val_change=True)

    def close(self) -> None:
        """Finish the run."""
        self.run.finish()
