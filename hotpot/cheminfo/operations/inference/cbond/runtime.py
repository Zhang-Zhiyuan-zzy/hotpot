"""Validated ONNX Runtime sessions for coordination-bond inference."""

from __future__ import annotations

from pathlib import Path

from hotpot.cheminfo.adapters.inference import cbond

from hotpot.cheminfo.operations.inference.artifacts import (
    ModelArtifact,
    load_manifest,
    verify_artifact,
)
from hotpot.cheminfo.operations.inference.artifacts.registry import ARTIFACTS


class CBondRuntime:
    def __init__(
        self,
        model_dir=None,
        device: str = "auto",
        verify: bool = True,
        model_source=None,
    ):
        pointer = ARTIFACTS["cbond"].manifest_path
        self.model_dir = ModelArtifact(
            pointer,
            "HOTPOT_CBOND_MODEL_DIR",
            model_dir=model_dir,
            source=model_source,
        ).resolve()
        self.manifest = load_manifest(self.model_dir / "manifest.json")
        if verify:
            verify_artifact(self.model_dir, self.manifest)
        self.requested_device = self._resolve_device(device)

        graph_path = self._model_path("graph")
        cbond_path = self._model_path("cbond")
        self.graph_session, self.cbond_session, self.device = cbond.create_sessions(
            graph_path, cbond_path, self.requested_device, device
        )

    @staticmethod
    def _resolve_device(device: str) -> str:
        return cbond.resolve_device(device)

    def _model_path(self, name: str) -> Path:
        entry = self.manifest["models"][name]
        return self.model_dir / entry["file"]

    def embed_graph(self, x, edge_index):
        return cbond.embed_graph(self.graph_session, x, edge_index)

    def predict(self, xg, padded_rings, rings_mask, cbond_index):
        return cbond.predict(
            self.cbond_session, xg, padded_rings, rings_mask, cbond_index
        )
