"""Resolve and verify MCA ONNX artifacts."""

from __future__ import annotations

from pathlib import Path

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

    def resolve(self, variant: str | None = None):
        selected = variant or self.manifest["default_variant"]
        entry = self.manifest["models"][selected]
        path = self.model_dir / entry["file"]
        if self.verify:
            verify_artifact(self.model_dir, self.manifest)
        return path, selected
