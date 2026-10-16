"""The logger protocol and a fan-out helper."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Protocol


class Logger(Protocol):
    """What the trainer calls. Implement these three methods and pass an instance."""

    def log_scalars(self, step: int, values: Mapping[str, float]) -> None:
        """Record numeric values at a global step (iteration or epoch, by key prefix)."""
        ...

    def log_config(self, config: Mapping[str, Any]) -> None:
        """Record the run configuration once."""
        ...

    def close(self) -> None:
        """Flush and release resources."""
        ...


class MultiLogger:
    """Forward every call to several loggers."""

    def __init__(self, loggers: Sequence[Logger]) -> None:
        self.loggers = list(loggers)

    def log_scalars(self, step: int, values: Mapping[str, float]) -> None:
        """See :class:`Logger`."""
        for lg in self.loggers:
            lg.log_scalars(step, values)

    def log_config(self, config: Mapping[str, Any]) -> None:
        """See :class:`Logger`."""
        for lg in self.loggers:
            lg.log_config(config)

    def close(self) -> None:
        """See :class:`Logger`."""
        for lg in self.loggers:
            lg.close()
