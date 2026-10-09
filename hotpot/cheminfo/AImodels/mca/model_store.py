"""Resolve and verify MCA ONNX artifacts."""

from __future__ import annotations

from pathlib import Path

import onnxruntime as ort

from hotpot.cheminfo.AImodels.artifacts import (
    ModelArtifact,
    load_manifest,
    verify_artifact,
)


class ModelStore:
    def __init__(self, model_dir=None, verify: bool = True, model_source=None):
        pointer = Path(__file__).with_name("models") / "manifest.json"
        self.model_dir = ModelArtifact(
            pointer,
            "HOTPOT_MCA_MODEL_DIR",
            model_dir=model_dir,
            source=model_source,
        ).resolve()
        self.manifest = load_manifest(self.model_dir / "manifest.json")
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
        selected = variant or self.manifest["default_variant"]
        entry = self.manifest["models"][selected]
        path = self.model_dir / entry["file"]
        if self.verify:
            verify_artifact(self.model_dir, self.manifest)
        providers = (
            ["CUDAExecutionProvider", "CPUExecutionProvider"]
            if resolved_device == "cuda"
            else ["CPUExecutionProvider"]
        )
        return path, selected, providers, resolved_device
