"""Validated ONNX Runtime sessions for coordination-bond inference."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import onnxruntime as ort

from hotpot.cheminfo.AImodels.artifacts import (
    ModelArtifact,
    load_manifest,
    verify_artifact,
)


MAX_RINGS_NUMS = 32
MAX_RINGS_SIZE = 64


def padding_rings(xg, rings_node_index, rings_node_nums):
    rings_nums = max(len(rings_node_nums), 1)
    rings_size = max(max(rings_node_nums, default=0), 1)
    if rings_nums > MAX_RINGS_NUMS or rings_size > MAX_RINGS_SIZE:
        raise ValueError(
            f"CBond ring dimensions ({rings_nums}, {rings_size}) exceed "
            f"the supported limits ({MAX_RINGS_NUMS}, {MAX_RINGS_SIZE})"
        )
    ring_vectors = xg[rings_node_index]
    ring_lengths = np.pad(rings_node_nums, (0, rings_nums - len(rings_node_nums)))[:, None]
    rings_mask = np.arange(rings_size) >= ring_lengths
    padded_rings = np.zeros((rings_nums, rings_size, xg.shape[-1]), dtype=xg.dtype)
    padded_rings[~rings_mask] = ring_vectors
    return padded_rings, rings_mask


class CBondRuntime:
    def __init__(
        self,
        model_dir=None,
        device: str = "auto",
        verify: bool = True,
        model_source=None,
    ):
        pointer = Path(__file__).with_name("onnx") / "manifest.json"
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
        providers = (
            ["CUDAExecutionProvider", "CPUExecutionProvider"]
            if self.requested_device == "cuda"
            else ["CPUExecutionProvider"]
        )
        options = ort.SessionOptions()
        options.log_severity_level = 3
        self.graph_session = ort.InferenceSession(
            str(graph_path), sess_options=options, providers=providers
        )
        self.cbond_session = ort.InferenceSession(
            str(cbond_path), sess_options=options, providers=providers
        )
        active_providers = set(self.graph_session.get_providers()) & set(
            self.cbond_session.get_providers()
        )
        if device == "cuda" and "CUDAExecutionProvider" not in active_providers:
            raise RuntimeError("The CUDA execution provider could not be initialized")
        self.device = "cuda" if "CUDAExecutionProvider" in active_providers else "cpu"

    @staticmethod
    def _resolve_device(device: str) -> str:
        if device == "auto":
            return "cuda" if "CUDAExecutionProvider" in ort.get_available_providers() else "cpu"
        if device not in {"cpu", "cuda"}:
            raise ValueError("device must be 'auto', 'cpu' or 'cuda'")
        if device == "cuda" and "CUDAExecutionProvider" not in ort.get_available_providers():
            raise RuntimeError("CUDAExecutionProvider is not available")
        return device

    def _model_path(self, name: str) -> Path:
        entry = self.manifest["models"][name]
        return self.model_dir / entry["file"]

    def embed_graph(self, x, edge_index):
        return self.graph_session.run(["xg"], {"x": x, "edge_index": edge_index})[0]

    def predict(self, xg, padded_rings, rings_mask, cbond_index):
        return self.cbond_session.run(
            ["cbond"],
            {
                "xg": xg,
                "padded_Xr": padded_rings,
                "rings_mask": rings_mask,
                "cbond_index": cbond_index,
            },
        )[0]
