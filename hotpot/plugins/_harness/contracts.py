"""Typed process and provenance records for native backends."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Optional, Tuple


__all__ = [
    "ArtifactProvenance",
    "NativeProcessResult",
    "ProcessProvenance",
    "ProcessRequest",
]


@dataclass(frozen=True)
class ProcessRequest:
    """Describe one native process invocation without backend semantics."""

    argv: Tuple[str, ...]
    cwd: Path
    env: Mapping[str, str]
    timeout_seconds: Optional[float] = None


@dataclass(frozen=True)
class NativeProcessResult:
    """Record the observable facts of a completed native process."""

    argv: Tuple[str, ...]
    cwd: Path
    return_code: int
    stdout: str
    stderr: str
    elapsed_seconds: float


@dataclass(frozen=True)
class ArtifactProvenance:
    """Identify one process artifact by absolute path and content digest."""

    path: Path
    sha256: str


@dataclass(frozen=True)
class ProcessProvenance:
    """Record executable, process, and artifact lineage facts."""

    executable: Path
    executable_sha256: str
    argv: Tuple[str, ...]
    cwd: Path
    return_code: int
    stdout: str
    stderr: str
    elapsed_seconds: float
    artifacts: Tuple[ArtifactProvenance, ...]
