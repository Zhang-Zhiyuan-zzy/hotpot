"""Lazy access to the native Open Babel rule planner."""

from __future__ import annotations

from importlib import import_module
from types import ModuleType


__all__ = ()


def _native_module() -> ModuleType:
    try:
        return import_module("hotpot.cheminfo.obWrappers._ob_rules")
    except ImportError as exc:
        raise ImportError(
            "the hotpot.cheminfo.obWrappers native extension is unavailable; "
            "install a compatible Hotpot wheel or rebuild Hotpot from source"
        ) from exc
