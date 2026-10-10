"""Narrow non-chemical primitives for external native programs."""

from .contracts import (
    ArtifactProvenance,
    NativeProcessResult,
    ProcessProvenance,
    ProcessRequest,
)
from .executable import ExecutableNotFoundError, resolve_executable
from .process import ProcessTimeoutError, run_process
from .provenance import build_process_provenance, sha256_file
from .workspace import isolated_workspace


__all__ = [
    "ArtifactProvenance",
    "ExecutableNotFoundError",
    "NativeProcessResult",
    "ProcessProvenance",
    "ProcessRequest",
    "ProcessTimeoutError",
    "build_process_provenance",
    "isolated_workspace",
    "resolve_executable",
    "run_process",
    "sha256_file",
]
