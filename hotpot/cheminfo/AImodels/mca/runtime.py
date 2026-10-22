"""Explicit CPU and CUDA ONNX Runtime sessions for all-atom MCA inference."""

from __future__ import annotations

import os
from contextlib import contextmanager
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Mapping

import numpy as np

from .all_atom_model import materialize_all_atom_model
from .model_store import ModelStore

ORT_DTYPES = {
    "tensor(float)": np.float32,
    "tensor(float16)": np.float16,
    "tensor(int32)": np.int32,
    "tensor(int64)": np.int64,
}


@lru_cache(maxsize=1)
def _onnxruntime():
    import onnxruntime

    return onnxruntime


def cuda_provider_available() -> bool:
    ort = _onnxruntime()
    return "CUDAExecutionProvider" in ort.get_available_providers()


def _default_cpu_threads() -> int:
    if hasattr(os, "sched_getaffinity"):
        available = len(os.sched_getaffinity(0))
    else:
        available = os.cpu_count() or 1
    return min(8, available)


def _session_options(intra_op_threads: int):
    ort = _onnxruntime()
    options = ort.SessionOptions()
    options.log_severity_level = 3
    options.intra_op_num_threads = intra_op_threads
    options.inter_op_num_threads = 1
    options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    return options


class MCARuntime:
    def __init__(
        self,
        session_path,
        *,
        backend: str,
        variant: str,
        manifest: Mapping[str, object],
        cuda_device_id: int | None = None,
        cpu_threads: int | None = None,
    ):
        self.variant = variant
        self.manifest = manifest
        if backend == "cpu":
            options = _session_options(
                _default_cpu_threads() if cpu_threads is None else cpu_threads
            )
            providers = ["CPUExecutionProvider"]
        elif backend == "cuda":
            if not cuda_provider_available():
                raise RuntimeError("CUDAExecutionProvider is not available")
            options = _session_options(1)
            providers = [
                ("CUDAExecutionProvider", {"device_id": cuda_device_id}),
                "CPUExecutionProvider",
            ]
        else:
            raise ValueError("backend must be 'cpu' or 'cuda'")

        ort = _onnxruntime()
        self.session = ort.InferenceSession(
            str(session_path), sess_options=options, providers=providers
        )
        if (
            backend == "cuda"
            and "CUDAExecutionProvider" not in self.session.get_providers()
        ):
            raise RuntimeError("The CUDA execution provider could not be initialized")
        self.input_dtypes = {
            input_.name: ORT_DTYPES[input_.type] for input_ in self.session.get_inputs()
        }
        if "atom_index" in self.input_dtypes:
            raise RuntimeError(
                "The MCA model could not be materialized with the required "
                "all-atom layout"
            )

    def predict(self, arrays, batch_size: int):
        total = arrays["atom_mask"].shape[0]
        predictions = []
        for start in range(0, total, batch_size):
            stop = min(start + batch_size, total)
            feed = {
                name: arrays[name][start:stop].astype(dtype, copy=False)
                for name, dtype in self.input_dtypes.items()
            }
            predictions.append(self.session.run(["mca_kj_mol"], feed)[0])
        output = np.concatenate(predictions, axis=0).astype(np.float64, copy=False)
        return output

    def close(self):
        self.session = None


@dataclass(frozen=True)
class ResolvedModelArtifact:
    session_path: Path
    variant: str
    manifest: Mapping[str, object]


@contextmanager
def resolved_model_artifact(
    *,
    model_dir=None,
    variant: str | None = None,
    verify_model: bool = True,
    model_source=None,
):
    """Resolve, verify, and materialize one model artifact for session creation."""

    store = ModelStore(
        model_dir=model_dir,
        verify=verify_model,
        model_source=model_source,
    )
    path, selected_variant = store.resolve(variant)
    external_data = store.manifest["models"][selected_variant].get("external_data")
    external_digest = (
        external_data["sha256"]
        if verify_model and external_data is not None
        else None
    )
    with materialize_all_atom_model(path, external_digest) as session_path:
        yield ResolvedModelArtifact(
            session_path=Path(session_path),
            variant=selected_variant,
            manifest=store.manifest,
        )


def create_cpu_runtime(
    *,
    model_dir=None,
    variant: str | None = None,
    verify_model: bool = True,
    model_source=None,
    cpu_threads: int | None = None,
) -> MCARuntime:
    with resolved_model_artifact(
        model_dir=model_dir,
        variant=variant,
        verify_model=verify_model,
        model_source=model_source,
    ) as artifact:
        return MCARuntime(
            artifact.session_path,
            backend="cpu",
            variant=artifact.variant,
            manifest=artifact.manifest,
            cpu_threads=cpu_threads,
        )


def create_cuda_runtime(
    *,
    model_dir=None,
    variant: str | None = None,
    verify_model: bool = True,
    model_source=None,
    device_id: int = 0,
) -> MCARuntime:
    with resolved_model_artifact(
        model_dir=model_dir,
        variant=variant,
        verify_model=verify_model,
        model_source=model_source,
    ) as artifact:
        return create_cuda_runtime_from_artifact(
            session_path=artifact.session_path,
            variant=artifact.variant,
            max_atoms=int(artifact.manifest["max_atoms"]),
            device_id=device_id,
        )


def create_cuda_runtime_from_artifact(
    *,
    session_path,
    variant: str,
    max_atoms: int,
    device_id: int,
) -> MCARuntime:
    """Create a CUDA session from an already resolved all-atom artifact."""

    return MCARuntime(
        session_path,
        backend="cuda",
        variant=variant,
        manifest={"max_atoms": max_atoms},
        cuda_device_id=device_id,
    )
