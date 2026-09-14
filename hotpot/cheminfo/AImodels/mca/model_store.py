"""Resolve and verify packaged ONNX artifacts."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import onnxruntime as ort


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _bundle_sha256(paths: list[Path]) -> str:
    digest = hashlib.sha256()
    for path in paths:
        digest.update(path.name.encode())
        digest.update(b"\0")
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


class ModelStore:
    def __init__(self, model_dir: str | os.PathLike | None = None, verify: bool = True):
        configured = model_dir or os.environ.get("HOTPOT_MCA_MODEL_DIR")
        self.model_dir = Path(configured) if configured else Path(__file__).with_name("models")
        self.manifest = json.loads((self.model_dir / "manifest.json").read_text(encoding="utf-8"))
        self.verify = verify

    @staticmethod
    def resolve_device(device: str) -> str:
        if device == "auto":
            return "cuda" if "CUDAExecutionProvider" in ort.get_available_providers() else "cpu"
        if device not in {"cpu", "cuda"}:
            raise ValueError("device must be 'auto', 'cpu' or 'cuda'")
        if device == "cuda" and "CUDAExecutionProvider" not in ort.get_available_providers():
            raise RuntimeError("CUDAExecutionProvider is not available")
        return device

    def resolve(self, device: str, variant: str | None):
        resolved_device = self.resolve_device(device)
        selected = variant or "fp16"
        entry = self.manifest["models"][selected]
        path = self.model_dir / entry["file"]
        if not path.is_file():
            raise FileNotFoundError(f"ONNX artifact not found: {path}")
        if self.verify:
            if _sha256(path) != entry["sha256"]:
                raise RuntimeError(f"SHA-256 mismatch for {path}")
            external = entry["external_data"]
            external_paths = sorted(self.model_dir.glob(external["glob"]), key=lambda item: item.name)
            if len(external_paths) != external["count"]:
                raise RuntimeError("Incomplete MCA ONNX external-data bundle")
            if _bundle_sha256(external_paths) != external["sha256"]:
                raise RuntimeError("SHA-256 mismatch for MCA ONNX external-data bundle")
        providers = (
            ["CUDAExecutionProvider", "CPUExecutionProvider"]
            if resolved_device == "cuda"
            else ["CPUExecutionProvider"]
        )
        return path, selected, providers, resolved_device
