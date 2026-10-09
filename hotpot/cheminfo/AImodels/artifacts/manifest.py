"""Load and verify versioned model manifests."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


def load_manifest(path: Path) -> dict:
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


def artifact_is_complete(model_dir: Path, manifest: dict) -> bool:
    for entry in manifest["models"].values():
        if not (model_dir / entry["file"]).is_file():
            return False
        external = entry.get("external_data")
        if external is not None:
            paths = tuple(model_dir.glob(external["glob"]))
            if len(paths) != external["count"]:
                return False
    return True


def verify_artifact(model_dir: Path, manifest: dict) -> None:
    for entry in manifest["models"].values():
        path = model_dir / entry["file"]
        if not path.is_file():
            raise FileNotFoundError(f"Model artifact not found: {path}")
        if sha256(path) != entry["sha256"]:
            raise RuntimeError(f"SHA-256 mismatch for {path}")

        external = entry.get("external_data")
        if external is None:
            continue
        paths = sorted(model_dir.glob(external["glob"]), key=lambda item: item.name)
        if len(paths) != external["count"]:
            raise RuntimeError(f"Incomplete external-data bundle in {model_dir}")
        if bundle_sha256(paths) != external["sha256"]:
            raise RuntimeError(f"SHA-256 mismatch for external-data bundle in {model_dir}")
