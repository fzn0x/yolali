"""CSV logger: one row per step, columns grow as new keys appear."""

from __future__ import annotations

import csv
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any


class CsvLogger:
    """Append rows to ``<dir>/<name>.csv``; keys seen later get new columns on rewrite.

    Rows are buffered in memory and the file rewritten on every call so the
    header always covers every column. Fine for the row counts of a
    training run; not meant for per-iteration logging at high frequency.
    """

    def __init__(self, directory: str | Path, name: str = "metrics") -> None:
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.path = self.directory / f"{name}.csv"
        self.rows: list[dict[str, float]] = []
        self.columns: list[str] = ["step"]

    def log_scalars(self, step: int, values: Mapping[str, float]) -> None:
        """Add a row and rewrite the file."""
        row: dict[str, float] = {"step": float(step), **{k: float(v) for k, v in values.items()}}
        for k in row:
            if k not in self.columns:
                self.columns.append(k)
        self.rows.append(row)
        with self.path.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=self.columns)
            w.writeheader()
            for r in self.rows:
                w.writerow({k: r.get(k, "") for k in self.columns})

    def log_config(self, config: Mapping[str, Any]) -> None:
        """Write ``config.json`` next to the csv."""
        (self.directory / "config.json").write_text(json.dumps(dict(config), indent=2, default=str))

    def close(self) -> None:
        """Nothing to release."""
