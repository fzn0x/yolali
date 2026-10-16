"""TensorBoard logger (optional extra ``tensorboard``)."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any


class TensorBoardLogger:
    """Write scalars with ``torch.utils.tensorboard.SummaryWriter``."""

    def __init__(self, directory: str | Path) -> None:
        try:
            from torch.utils.tensorboard import SummaryWriter  # noqa: PLC0415 - optional dependency
        except ImportError as exc:
            raise ImportError("TensorBoardLogger needs `uv sync --extra tensorboard`") from exc
        self.writer = SummaryWriter(str(directory))

    def log_scalars(self, step: int, values: Mapping[str, float]) -> None:
        """See :class:`~yolali.train.loggers.Logger`."""
        for k, v in values.items():
            self.writer.add_scalar(k, v, step)

    def log_config(self, config: Mapping[str, Any]) -> None:
        """Store the config as text."""
        self.writer.add_text("config", json.dumps(dict(config), indent=2, default=str))

    def close(self) -> None:
        """Flush and close the writer."""
        self.writer.close()
