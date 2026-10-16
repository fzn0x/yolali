"""Everything needed to resume a run bit-for-bit."""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch

CHECKPOINT_VERSION = 1


@dataclass
class RNGState:
    """Random generator states the trainer draws from.

    NumPy is absent on purpose: the library never touches the legacy global
    NumPy generator; every NumPy draw uses an explicit ``Generator`` seeded
    from ``(seed, epoch, index)``.
    """

    python: object
    torch_cpu: torch.Tensor
    cuda: list[torch.Tensor] = field(default_factory=list[torch.Tensor])

    @classmethod
    def capture(cls) -> RNGState:
        """Snapshot the current states."""
        cuda = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []
        return cls(random.getstate(), torch.get_rng_state(), cuda)

    def restore(self) -> None:
        """Put the states back."""
        random.setstate(self.python)  # pyright: ignore[reportArgumentType] - opaque tuple from getstate
        torch.set_rng_state(self.torch_cpu)
        if self.cuda and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(self.cuda)


@dataclass
class Checkpoint:
    """One training snapshot. Tensors are on CPU."""

    model: dict[str, torch.Tensor]
    epoch: int  # last completed epoch
    optimizer: dict[str, Any] | None = None
    ema: dict[str, Any] | None = None
    scaler: dict[str, Any] | None = None
    rng: RNGState | None = None
    best_fitness: float = -1.0
    config: dict[str, Any] = field(default_factory=dict[str, Any])
    model_config: dict[str, Any] = field(default_factory=dict[str, Any])
    version: int = CHECKPOINT_VERSION


def save_checkpoint(ckpt: Checkpoint, path: str | Path) -> None:
    """Write atomically (temp file + rename)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(ckpt, tmp)
    tmp.replace(path)


def load_checkpoint(path: str | Path) -> Checkpoint:
    """Read a :class:`Checkpoint`; raises on anything else."""
    obj: Any = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(obj, Checkpoint):
        raise ValueError(f"{path} is not a yolali Checkpoint (got {type(obj).__name__})")
    return obj
