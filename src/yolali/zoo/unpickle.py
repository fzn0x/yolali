"""Read a pickled ``nn.Module`` checkpoint without importing its source code.

Official YOLOv9 checkpoints pickle whole ``DetectionModel`` objects whose
classes live in the reference repository (``models.*``, ``utils.*``). Those
files are GPL-3.0 and are not a dependency of this library, so instead of
importing them we hand the unpickler a stub class for every foreign name.
``torch.nn.Module.__setstate__`` semantics are just "update ``__dict__``",
which is all a stub needs to hold the ``_parameters`` / ``_buffers`` /
``_modules`` trees that make up a state dict.
"""

from __future__ import annotations

import pickle
import types
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

_FOREIGN_PREFIXES = ("models.", "utils.")


class _StubModule:
    """Stand-in for any class from the reference repository."""

    def __init__(self, *args: object, **kwargs: object) -> None:
        pass

    def __setstate__(self, state: object) -> None:
        if isinstance(state, dict):
            self.__dict__.update(state)  # pyright: ignore[reportUnknownArgumentType] - pickle state


class _Unpickler(pickle.Unpickler):
    def find_class(self, module: str, name: str) -> Any:  # noqa: ANN401 - pickle protocol signature
        if module.startswith(_FOREIGN_PREFIXES):
            return type(name, (_StubModule,), {"__module__": module})
        return super().find_class(module, name)


def _walk(obj: object, prefix: str, out: dict[str, Tensor]) -> None:
    """Collect ``_parameters`` and ``_buffers`` from a module-like tree."""
    params: Mapping[str, Any] = getattr(obj, "_parameters", {}) or {}
    buffers: Mapping[str, Any] = getattr(obj, "_buffers", {}) or {}
    children: Mapping[str, Any] = getattr(obj, "_modules", {}) or {}
    for name, value in list(params.items()) + list(buffers.items()):
        if isinstance(value, Tensor):
            out[prefix + name] = value.detach().clone()
    for name, child in children.items():
        if child is not None:
            _walk(child, f"{prefix}{name}.", out)


def load_foreign_state_dict(path: str | Path, prefer_ema: bool = True) -> dict[str, Tensor]:
    """Return ``{name: tensor}`` for the model pickled in a reference checkpoint.

    Args:
        path: ``.pt`` file saved by the reference training code.
        prefer_ema: use the ``ema`` entry when it is present and not ``None``.
    """
    pickle_module = types.ModuleType("yolali_stub_pickle")
    pickle_module.Unpickler = _Unpickler  # pyright: ignore[reportAttributeAccessIssue] - duck-typed pickle module
    pickle_module.load = pickle.load  # pyright: ignore[reportAttributeAccessIssue] - as above
    ckpt: Any = torch.load(
        path, map_location="cpu", pickle_module=pickle_module, weights_only=False
    )
    if not isinstance(ckpt, dict):
        raise ValueError(f"{path}: expected a checkpoint dict, got {type(ckpt).__name__}")
    ckpt_dict: dict[str, Any] = ckpt  # pyright: ignore[reportUnknownVariableType] - untyped pickle
    model = ckpt_dict.get("ema") if prefer_ema else None
    if model is None:
        model = ckpt_dict.get("model")
    if model is None:
        raise ValueError(f"{path}: checkpoint has neither 'ema' nor 'model'")
    state: dict[str, Tensor] = {}
    _walk(model, "", state)
    if not state:
        raise ValueError(f"{path}: no tensors found in the pickled model")
    return {k: v.float() if v.is_floating_point() else v for k, v in state.items()}
