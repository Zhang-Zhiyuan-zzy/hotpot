"""Ordered logical-GPU hardware memory observations."""
from __future__ import annotations
import os
import subprocess
from dataclasses import dataclass
from typing import Callable, Sequence
MIB = 1024**2


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
