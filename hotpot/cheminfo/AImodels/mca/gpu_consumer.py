"""One-process-per-GPU MCA inference consumer."""

from __future__ import annotations

import traceback
from dataclasses import dataclass
from typing import Mapping

import numpy as np

from .featurizer import collate_molecule_rows
from .resources import pin_current_process
from .runtime import create_cuda_runtime_from_artifact


@dataclass(frozen=True)
class GpuConsumerConfig:
    session_path: str
    variant: str
    max_atoms: int
    batch_size: int


@dataclass(frozen=True)
class GpuBatchRequest:
    request_id: int
    batch_ordinal: int
    pad_width: int
    molecule_indices: tuple[int, ...]
    features: tuple[Mapping[str, np.ndarray], ...]


@dataclass(frozen=True)
class GpuConsumerReady:
    gpu_index: int
    variant: str
    max_atoms: int


@dataclass(frozen=True)
class GpuBatchResult:
    request_id: int
    batch_ordinal: int
    gpu_index: int
    molecule_indices: tuple[int, ...]
    pad_width: int
    predictions: tuple[np.ndarray, ...]


@dataclass(frozen=True)
class GpuConsumerFailure:
    gpu_index: int
    request_id: int | None
    batch_ordinal: int | None
    exception_type: str
    message: str
    traceback_text: str


def gpu_consumer_main(
    gpu_index: int,
    cpu_ids: tuple[int, ...],
    config: GpuConsumerConfig,
    request_queue,
    result_queue,
) -> None:
    """Own one CUDA session and execute requests until receiving ``None``."""

    runtime = None
    try:
        try:
            pin_current_process(cpu_ids)
            runtime = create_cuda_runtime_from_artifact(
                session_path=config.session_path,
                variant=config.variant,
                max_atoms=config.max_atoms,
                device_id=gpu_index,
            )
            result_queue.put(
                GpuConsumerReady(
                    gpu_index=gpu_index,
                    variant=runtime.variant,
                    max_atoms=config.max_atoms,
                )
            )
        except BaseException as exc:
            result_queue.put(
                GpuConsumerFailure(
                    gpu_index=gpu_index,
                    request_id=None,
                    batch_ordinal=None,
                    exception_type=type(exc).__name__,
                    message=str(exc),
                    traceback_text=traceback.format_exc(),
                )
            )
            return

        while True:
            request = request_queue.get()
            if request is None:
                return
            try:
                arrays = collate_molecule_rows(
                    request.features,
                    range(len(request.features)),
                    max_atoms=request.pad_width,
                )
                rows = runtime.predict(arrays, config.batch_size)
                result_queue.put(
                    GpuBatchResult(
                        request_id=request.request_id,
                        batch_ordinal=request.batch_ordinal,
                        gpu_index=gpu_index,
                        molecule_indices=request.molecule_indices,
                        pad_width=request.pad_width,
                        predictions=tuple(
                            np.asarray(row, dtype=np.float64) for row in rows
                        ),
                    )
                )
            except BaseException as exc:
                result_queue.put(
                    GpuConsumerFailure(
                        gpu_index=gpu_index,
                        request_id=request.request_id,
                        batch_ordinal=request.batch_ordinal,
                        exception_type=type(exc).__name__,
                        message=str(exc),
                        traceback_text=traceback.format_exc(),
                    )
                )
    finally:
        if runtime is not None:
            runtime.close()
