"""Load and verify versioned model manifests."""

from __future__ import annotations

from pathlib import Path

from ....adapters.inference import artifacts as _effects


def load_manifest(path: Path) -> dict:
    return _effects.read_json(path)


def sha256(path: Path) -> str:
    return _effects.sha256(path)


def bundle_sha256(paths: list[Path]) -> str:
    return _effects.bundle_sha256(paths)


def artifact_is_complete(model_dir: Path, manifest: dict) -> bool:
    for entry in manifest["models"].values():
        if not _effects.is_file(model_dir / entry["file"]):
            return False
        external = entry.get("external_data")
        if external is not None:
            _, count = _effects.external_files(model_dir, external["glob"])
            if count != external["count"]:
                return False
    return True


def verify_artifact(model_dir: Path, manifest: dict) -> None:
    for entry in manifest["models"].values():
        path = model_dir / entry["file"]
        if not _effects.is_file(path):
            raise FileNotFoundError(f"Model artifact not found: {path}")
        if sha256(path) != entry["sha256"]:
            raise RuntimeError(f"SHA-256 mismatch for {path}")

        external = entry.get("external_data")
        if external is None:
            continue
        paths, count = _effects.external_files(model_dir, external["glob"])
        paths = sorted(paths, key=lambda item: item.name)
        if count != external["count"]:
            raise RuntimeError(f"Incomplete external-data bundle in {model_dir}")
        if bundle_sha256(paths) != external["sha256"]:
            raise RuntimeError(f"SHA-256 mismatch for external-data bundle in {model_dir}")
