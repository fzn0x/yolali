"""Label cache keyed by a content fingerprint.

Reading thousands of label files and image headers takes a while; the result
is stored next to the data as ``<name>.cache``. The fingerprint covers every
image path, its size and mtime, plus a reader version, so any change to the
dataset (or to the reader) rebuilds the cache automatically.
"""

from __future__ import annotations

import hashlib
import pickle
from collections.abc import Callable, Sequence
from dataclasses import asdict
from pathlib import Path
from typing import cast

import numpy as np

from yolali.data.formats.record import LabelRecord

CACHE_VERSION = 1


def fingerprint(paths: Sequence[Path], extra: str = "") -> str:
    """SHA-1 over sorted ``(path, size, mtime_ns)`` triples and ``extra``."""
    h = hashlib.sha1(f"v{CACHE_VERSION}|{extra}".encode())
    for p in sorted(paths):
        st = p.stat()
        h.update(f"{p}|{st.st_size}|{st.st_mtime_ns}".encode())
    return h.hexdigest()


class LabelCache:
    """Load ``list[LabelRecord]`` from ``cache_path`` when its fingerprint matches, else rebuild."""

    def __init__(self, cache_path: str | Path) -> None:
        self.cache_path = Path(cache_path)

    def load_or_build(
        self, paths: Sequence[Path], build: Callable[[], list[LabelRecord]], extra: str = ""
    ) -> list[LabelRecord]:
        """Return cached records when the fingerprint matches, else ``build()`` and save."""
        fp = fingerprint(paths, extra)
        cached = self._read(fp)
        if cached is not None:
            return cached
        records = build()
        self._write(fp, records)
        return records

    def _read(self, fp: str) -> list[LabelRecord] | None:
        if not self.cache_path.exists():
            return None
        try:
            with self.cache_path.open("rb") as f:
                payload = pickle.load(f)  # our own file, written by _write below
        except (OSError, pickle.UnpicklingError, EOFError):
            return None
        if not isinstance(payload, dict) or payload.get("fingerprint") != fp:
            return None
        rows = cast(list[dict[str, object]], payload["records"])
        return [
            LabelRecord(
                path=str(r["path"]),
                width=int(r["width"]),  # pyright: ignore[reportArgumentType] - stored as int
                height=int(r["height"]),  # pyright: ignore[reportArgumentType] - stored as int
                boxes=np.asarray(r["boxes"], dtype=np.float32),
                classes=np.asarray(r["classes"], dtype=np.int64),
                crowd=np.asarray(r["crowd"], dtype=np.bool_),
            )
            for r in rows
        ]

    def _write(self, fp: str, records: list[LabelRecord]) -> None:
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"fingerprint": fp, "records": [asdict(r) for r in records]}
        tmp = self.cache_path.with_suffix(".cache.tmp")
        with tmp.open("wb") as f:
            pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
        tmp.replace(self.cache_path)
