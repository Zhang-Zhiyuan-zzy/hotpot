"""Standalone and Hotpot-compatible MCA inference package."""

from __future__ import annotations


__all__ = [
    "AtomPrediction",
    "MCACpuPredictor",
    "MCAGpuOutOfMemoryError",
    "MCAGpuPredictor",
    "MoleculeGraph",
    "MoleculePrediction",
    "SitePrediction",
    "predict_mca",
    "predict_mca_cpu",
    "predict_mca_gpu",
]


def __getattr__(name):
    if name in {
        "MCACpuPredictor",
        "MCAGpuPredictor",
        "predict_mca",
        "predict_mca_cpu",
        "predict_mca_gpu",
    }:
        from . import api

        return getattr(api, name)
    if name == "MCAGpuOutOfMemoryError":
        from .gpu_memory import MCAGpuOutOfMemoryError

        return MCAGpuOutOfMemoryError
    if name == "MoleculeGraph":
        from .graph_adapter import MoleculeGraph

        return MoleculeGraph
    if name in {"AtomPrediction", "MoleculePrediction", "SitePrediction"}:
        from . import result_types

        return getattr(result_types, name)
    raise AttributeError(name)
