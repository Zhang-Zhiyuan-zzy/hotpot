"""Validated ONNX Runtime sessions for coordination-bond inference."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import numpy as np
import onnxruntime as ort


MAX_RINGS_NUMS = 32
MAX_RINGS_SIZE = 64


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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
        model_dir: str | os.PathLike | None = None,
        device: str = "auto",
        verify: bool = True,
    ):
        configured = model_dir or os.environ.get("HOTPOT_CBOND_MODEL_DIR")
        self.model_dir = Path(configured) if configured else Path(__file__).with_name("onnx")
        self.manifest = json.loads((self.model_dir / "manifest.json").read_text(encoding="utf-8"))
        self.requested_device = self._resolve_device(device)

        graph_path = self._model_path("graph", verify)
        cbond_path = self._model_path("cbond", verify)
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

    def _model_path(self, name: str, verify: bool) -> Path:
        entry = self.manifest["models"][name]
        path = self.model_dir / entry["file"]
        if not path.is_file():
            raise FileNotFoundError(f"CBond ONNX artifact not found: {path}")
        if verify and _sha256(path) != entry["sha256"]:
            raise RuntimeError(f"SHA-256 mismatch for {path}")
        return path

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
