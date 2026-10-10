"""Resolve native executables with explicit, environment, and PATH precedence."""

from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import Mapping, Optional, Union


__all__ = ["ExecutableNotFoundError", "resolve_executable"]


class ExecutableNotFoundError(FileNotFoundError):
    """Raised when a requested native executable cannot be resolved."""


_ExecutablePath = Union[str, os.PathLike[str]]


def _resolve_candidate(
    candidate: _ExecutablePath,
    search_path: str,
) -> Optional[Path]:
    candidate_path = Path(candidate).expanduser()
    if candidate_path.is_file() and os.access(candidate_path, os.X_OK):
        return candidate_path.resolve()

    located = shutil.which(os.fspath(candidate), path=search_path)
    if located is None:
        return None
    return Path(located).resolve()


def _require_candidate(
    candidate: _ExecutablePath,
    *,
    source: str,
    search_path: str,
) -> Path:
    resolved = _resolve_candidate(candidate, search_path)
    if resolved is None:
        raise ExecutableNotFoundError(
            f"{source} executable could not be resolved: {os.fspath(candidate)!r}"
        )
    return resolved


def resolve_executable(
    name: str,
    *,
    explicit: Optional[_ExecutablePath] = None,
    env_var: Optional[str] = None,
    env: Optional[Mapping[str, str]] = None,
) -> Path:
    """Resolve an executable as ``explicit > env_var > PATH``."""

    environment = os.environ if env is None else env
    search_path = environment.get("PATH", "")

    if explicit is not None:
        return _require_candidate(
            explicit,
            source="explicit",
            search_path=search_path,
        )

    if env_var is not None and env_var in environment:
        return _require_candidate(
            environment[env_var],
            source=env_var,
            search_path=search_path,
        )

    resolved = shutil.which(name, path=search_path)
    if resolved is None:
        raise ExecutableNotFoundError(f"executable was not found on PATH: {name!r}")
    return Path(resolved).resolve()
