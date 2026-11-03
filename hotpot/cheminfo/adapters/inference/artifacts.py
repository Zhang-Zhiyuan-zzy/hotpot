"""Filesystem, digest and pinned-download effects for inference artifacts."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from huggingface_hub import snapshot_download
from huggingface_hub.errors import LocalEntryNotFoundError


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def bundle_sha256(paths: list[Path]) -> str:
    digest = hashlib.sha256()
    for path in paths:
        digest.update(path.name.encode())
        digest.update(b"\0")
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


def is_file(path: Path) -> bool:
    return path.is_file()


def external_files(directory: Path, pattern: str) -> tuple[tuple[Path, ...], int]:
    paths = tuple(directory.glob(pattern))
    return paths, len(paths)


def download_snapshot(
    *,
    repo_id: str,
    revision: str,
    allow_patterns: list[str],
    cache_dir: Path,
    local_dir: Path,
    local_files_only: bool,
):
    return snapshot_download(
        repo_id=repo_id,
        revision=revision,
        allow_patterns=allow_patterns,
        cache_dir=cache_dir,
        local_dir=local_dir,
        local_files_only=local_files_only,
    )
