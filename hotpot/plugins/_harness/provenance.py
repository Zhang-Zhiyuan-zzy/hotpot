"""Content-addressed process and artifact provenance."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Iterable

from .contracts import ArtifactProvenance, NativeProcessResult, ProcessProvenance


__all__ = ["build_process_provenance", "sha256_file"]


def sha256_file(path: Path) -> str:
    """Return the hexadecimal SHA-256 digest of a file's bytes."""

    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def build_process_provenance(
    *,
    executable: Path,
    result: NativeProcessResult,
    artifacts: Iterable[Path],
) -> ProcessProvenance:
    """Build immutable lineage records from one completed process."""

    executable_path = executable.resolve()
    artifact_records = tuple(
        ArtifactProvenance(path=path.resolve(), sha256=sha256_file(path))
        for path in artifacts
    )
    return ProcessProvenance(
        executable=executable_path,
        executable_sha256=sha256_file(executable_path),
        argv=result.argv,
        cwd=result.cwd,
        return_code=result.return_code,
        stdout=result.stdout,
        stderr=result.stderr,
        elapsed_seconds=result.elapsed_seconds,
        artifacts=artifact_records,
    )
