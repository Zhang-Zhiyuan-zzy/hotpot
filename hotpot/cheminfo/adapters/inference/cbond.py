"""ONNX sessions and tensor preparation for resolved CBond models."""

from __future__ import annotations

import numpy as np
import onnxruntime as ort


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


def resolve_device(device: str) -> str:
    if device == "auto":
        return "cuda" if "CUDAExecutionProvider" in ort.get_available_providers() else "cpu"
    if device not in {"cpu", "cuda"}:
        raise ValueError("device must be 'auto', 'cpu' or 'cuda'")
    if device == "cuda" and "CUDAExecutionProvider" not in ort.get_available_providers():
        raise RuntimeError("CUDAExecutionProvider is not available")
    return device


def create_sessions(graph_path, cbond_path, requested_device, device):
    providers = (
        ["CUDAExecutionProvider", "CPUExecutionProvider"]
        if requested_device == "cuda"
        else ["CPUExecutionProvider"]
    )
    options = ort.SessionOptions()
    options.log_severity_level = 3
    graph_session = ort.InferenceSession(
        str(graph_path), sess_options=options, providers=providers
    )
    cbond_session = ort.InferenceSession(
        str(cbond_path), sess_options=options, providers=providers
    )
    active_providers = set(graph_session.get_providers()) & set(
        cbond_session.get_providers()
    )
    if device == "cuda" and "CUDAExecutionProvider" not in active_providers:
        raise RuntimeError("The CUDA execution provider could not be initialized")
    active_device = "cuda" if "CUDAExecutionProvider" in active_providers else "cpu"
    return graph_session, cbond_session, active_device


def embed_graph(session, x, edge_index):
    return session.run(["xg"], {"x": x, "edge_index": edge_index})[0]


def predict(session, xg, padded_rings, rings_mask, cbond_index):
    return session.run(
        ["cbond"],
        {
            "xg": xg,
            "padded_Xr": padded_rings,
            "rings_mask": rings_mask,
            "cbond_index": cbond_index,
        },
    )[0]
