"""Standalone and Hotpot-compatible MCA inference package."""

from __future__ import annotations


__all__ = [
    "mca",
    "AtomPrediction",
    "MCACpuPredictor",
    "MCAGpuOutOfMemoryError",
    "MCAGpuPredictor",
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
    if name in {"AtomPrediction", "MoleculePrediction", "SitePrediction"}:
        from . import results

        return getattr(results, name)
    if name == "mca":
        from .attachment import mca

        return mca
    raise AttributeError(name)
