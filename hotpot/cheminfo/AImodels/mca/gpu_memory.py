"""GPU-memory batch selection and CUDA OOM reporting for MCA."""

from __future__ import annotations

import os
import subprocess
from dataclasses import dataclass
from typing import Callable, Sequence


GIB = 1024**3
MIB = 1024**2
SUPPORTED_GPU_BATCH_SIZES = (64, 128, 256, 512, 1024, 2048)


@dataclass(frozen=True)
class GPUDeviceMemory:
    gpu_index: int
    total_bytes: int
    free_bytes: int | None = None


def _visible_gpu_selector(gpu_index: int) -> str:
    visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible_devices is None:
        return str(gpu_index)
    selectors = tuple(
        selector.strip()
        for selector in visible_devices.split(",")
        if selector.strip()
    )
    if gpu_index >= len(selectors):
        raise RuntimeError(
            f"Logical GPU index {gpu_index} is not present in CUDA_VISIBLE_DEVICES"
        )
    return selectors[gpu_index]


def parse_nvidia_smi_memory(output: str, gpu_index: int) -> GPUDeviceMemory:
    """Parse one ``memory.total,memory.free`` nvidia-smi CSV row."""

    rows = [row.strip() for row in output.splitlines() if row.strip()]
    if len(rows) != 1:
        raise RuntimeError(
            f"Expected one nvidia-smi memory row for GPU {gpu_index}, got {len(rows)}"
        )
    fields = [field.strip() for field in rows[0].split(",")]
    if len(fields) != 2:
        raise RuntimeError(f"Malformed nvidia-smi memory row for GPU {gpu_index}")
    try:
        total_mib, free_mib = (int(field) for field in fields)
    except ValueError as error:
        raise RuntimeError(
            f"Malformed nvidia-smi memory values for GPU {gpu_index}: {rows[0]}"
        ) from error
    return GPUDeviceMemory(
        gpu_index=gpu_index,
        total_bytes=total_mib * MIB,
        free_bytes=free_mib * MIB,
    )


def gpu_memory_bytes(gpu_index: int) -> GPUDeviceMemory:
    """Query total and currently free memory for one logical GPU."""

    selector = _visible_gpu_selector(gpu_index)
    completed = subprocess.run(
        (
            "nvidia-smi",
            f"--id={selector}",
            "--query-gpu=memory.total,memory.free",
            "--format=csv,noheader,nounits",
        ),
        check=True,
        capture_output=True,
        text=True,
    )
    return parse_nvidia_smi_memory(completed.stdout, gpu_index)


def gpu_total_memory_bytes(gpu_index: int) -> int:
    """Return total memory for one logical GPU in bytes."""

    return gpu_memory_bytes(gpu_index).total_bytes


def gpu_total_memories_bytes(
    gpu_indices: Sequence[int],
    memory_query: Callable[[int], int] | None = None,
) -> tuple[int, ...]:
    """Query each selected GPU once, preserving the requested logical order."""

    query = gpu_total_memory_bytes if memory_query is None else memory_query
    return tuple(query(gpu_index) for gpu_index in gpu_indices)


def automatic_gpu_batch_size(total_memory_bytes: int) -> int:
    """Choose the approved MCA batch tier from total (not free) VRAM."""

    if total_memory_bytes < 8 * GIB:
        return 64
    if total_memory_bytes < 16 * GIB:
        return 128
    if total_memory_bytes < 24 * GIB:
        return 256
    if total_memory_bytes < 40 * GIB:
        return 512
    if total_memory_bytes < 80 * GIB:
        return 1024
    return 2048


def validate_gpu_batch_size(batch_size: int) -> int:
    if (
        isinstance(batch_size, bool)
        or not isinstance(batch_size, int)
        or batch_size not in SUPPORTED_GPU_BATCH_SIZES
    ):
        supported = ", ".join(str(value) for value in SUPPORTED_GPU_BATCH_SIZES)
        raise ValueError(f"gpu_batch_size must be one of: {supported}")
    return batch_size


def resolve_gpu_batch_size(
    total_memory_bytes: Sequence[int],
    requested_batch_size: int | None = None,
) -> int:
    """Resolve one global batch, using the smallest heterogeneous-card tier."""

    if requested_batch_size is not None:
        return validate_gpu_batch_size(requested_batch_size)
    memories = tuple(total_memory_bytes)
    if not memories:
        raise ValueError("At least one GPU memory value is required")
    return min(automatic_gpu_batch_size(total) for total in memories)


def next_lower_gpu_batch_size(batch_size: int) -> int | None:
    lower = [value for value in SUPPORTED_GPU_BATCH_SIZES if value < batch_size]
    return lower[-1] if lower else None


class MCAGpuOutOfMemoryError(RuntimeError):
    """A CUDA allocation failed during MCA GPU inference."""

    def __init__(
        self,
        *,
        gpu_index: int,
        batch_size: int,
        operation: str,
        total_memory_bytes: int | None = None,
        free_memory_bytes: int | None = None,
        variant: str | None = None,
    ):
        self.gpu_index = gpu_index
        self.batch_size = batch_size
        self.next_batch_size = next_lower_gpu_batch_size(batch_size)
        self.operation = operation
        self.total_memory_bytes = total_memory_bytes
        self.free_memory_bytes = free_memory_bytes
        self.variant = variant

        details = [
            f"MCA GPU out of memory on logical GPU {gpu_index}",
            f"during {operation}",
            f"with gpu_batch_size={batch_size}",
        ]
        if variant is not None:
            details.append(f"model_variant={variant}")
        if total_memory_bytes is not None:
            details.append(f"total_vram={total_memory_bytes / GIB:.2f} GiB")
        if free_memory_bytes is not None:
            details.append(f"free_vram={free_memory_bytes / GIB:.2f} GiB")
        message = "; ".join(details) + ". "
        if self.next_batch_size is None:
            message += (
                "gpu_batch_size=64 is already the lowest supported batch tier; "
                "no automatic retry or CPU fallback was attempted."
            )
        else:
            message += (
                f"Retry explicitly with gpu_batch_size={self.next_batch_size}; "
                "no automatic retry or CPU fallback was attempted."
            )
        super().__init__(message)


_CUDA_OOM_MARKERS = (
    "cuda out of memory",
    "cuda_error_out_of_memory",
    "cudaerroroutofmemory",
    "cudnn_status_alloc_failed",
    "cublas_status_alloc_failed",
    "hip_error_out_of_memory",
    "failed to allocate memory for requested buffer",
    "bfc arena ran out of memory",
)


def is_cuda_out_of_memory(error: BaseException) -> bool:
    """Recognize CUDA/ORT allocation failures without masking other errors."""

    current: BaseException | None = error
    visited = set()
    while current is not None and id(current) not in visited:
        visited.add(id(current))
        message = str(current).lower()
        if any(marker in message for marker in _CUDA_OOM_MARKERS):
            return True
        if (
            ("cuda" in message or "cudaexecutionprovider" in message)
            and ("out of memory" in message or "failed to allocate" in message)
        ):
            return True
        current = current.__cause__ or current.__context__
    return False
