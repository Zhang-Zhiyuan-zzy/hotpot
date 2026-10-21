"""Deterministic CPU and GPU resource planning for MCA inference."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Literal, Sequence


DEFAULT_SITE_CPU_PER_GPU = 9
DEFAULT_INFER_CPU_PER_GPU = 7


class InsufficientCPUAffinityError(RuntimeError):
    """The process affinity cannot supply the configured MCA workers."""

    def __init__(
        self,
        *,
        gpu_count: int,
        site_cpu_per_gpu: int,
        infer_cpu_per_gpu: int,
        required_cpu_count: int,
        available_cpu_count: int,
    ):
        self.gpu_count = gpu_count
        self.site_cpu_per_gpu = site_cpu_per_gpu
        self.infer_cpu_per_gpu = infer_cpu_per_gpu
        self.required_cpu_count = required_cpu_count
        self.available_cpu_count = available_cpu_count
        super().__init__(
            "MCA GPU inference requires "
            f"{required_cpu_count} CPUs for {gpu_count} GPU(s) "
            f"({site_cpu_per_gpu} site + {infer_cpu_per_gpu} inference CPUs "
            f"per GPU), but the current process affinity exposes only "
            f"{available_cpu_count} CPUs"
        )


@dataclass(frozen=True)
class CPUWorkerAssignment:
    """One CPU worker pinned to one affinity-visible CPU."""

    stage: Literal["site", "infer"]
    gpu_index: int
    stage_worker_index: int
    global_worker_index: int
    cpu_id: int


@dataclass(frozen=True)
class GPUCPUAssignment:
    """The complete CPU block associated with one GPU consumer."""

    gpu_index: int
    cpu_ids: tuple[int, ...]
    site_cpu_ids: tuple[int, ...]
    infer_cpu_ids: tuple[int, ...]


@dataclass(frozen=True)
class GPUResourcePlan:
    """Fully resolved GPU selection and CPU-affinity allocation."""

    gpu_indices: tuple[int, ...]
    available_cpu_ids: tuple[int, ...]
    site_cpu_per_gpu: int
    infer_cpu_per_gpu: int
    gpu_cpu_assignments: tuple[GPUCPUAssignment, ...]
    site_worker_assignments: tuple[CPUWorkerAssignment, ...]
    infer_worker_assignments: tuple[CPUWorkerAssignment, ...]

    @property
    def gpu_count(self) -> int:
        return len(self.gpu_indices)

    @property
    def required_cpu_count(self) -> int:
        return self.gpu_count * (
            self.site_cpu_per_gpu + self.infer_cpu_per_gpu
        )

    @property
    def assigned_cpu_ids(self) -> tuple[int, ...]:
        return tuple(
            cpu_id
            for assignment in self.gpu_cpu_assignments
            for cpu_id in assignment.cpu_ids
        )


def _positive_integer(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def resolve_gpu_indices(
    *,
    gpu_count: int | None = None,
    gpu_indices: Sequence[int] | None = None,
) -> tuple[int, ...]:
    """Resolve the public count/indices choice to ordered logical GPU IDs."""

    if gpu_count is not None and gpu_indices is not None:
        raise ValueError("gpu_count and gpu_indices are mutually exclusive")
    if gpu_indices is None:
        count = 1 if gpu_count is None else _positive_integer(gpu_count, "gpu_count")
        return tuple(range(count))

    indices = tuple(gpu_indices)
    if not indices:
        raise ValueError("gpu_indices must not be empty")
    if any(
        isinstance(index, bool) or not isinstance(index, int) or index < 0
        for index in indices
    ):
        raise ValueError("gpu_indices must contain non-negative integers")
    if len(set(indices)) != len(indices):
        raise ValueError("gpu_indices must not contain duplicates")
    return indices


def current_cpu_affinity() -> tuple[int, ...]:
    """Return the CPUs made available by the enclosing cpuset/cgroup."""

    return tuple(sorted(os.sched_getaffinity(0)))


def build_gpu_resource_plan(
    *,
    gpu_count: int | None = None,
    gpu_indices: Sequence[int] | None = None,
    site_cpu_per_gpu: int = DEFAULT_SITE_CPU_PER_GPU,
    infer_cpu_per_gpu: int = DEFAULT_INFER_CPU_PER_GPU,
    cpu_affinity: Sequence[int] | None = None,
) -> GPUResourcePlan:
    """Allocate stable, non-overlapping CPU blocks for the selected GPUs."""

    indices = resolve_gpu_indices(gpu_count=gpu_count, gpu_indices=gpu_indices)
    site_count = _positive_integer(site_cpu_per_gpu, "site_cpu_per_gpu")
    infer_count = _positive_integer(infer_cpu_per_gpu, "infer_cpu_per_gpu")
    affinity = (
        current_cpu_affinity()
        if cpu_affinity is None
        else tuple(sorted(cpu_affinity))
    )
    if len(set(affinity)) != len(affinity):
        raise ValueError("cpu_affinity must not contain duplicates")

    cpu_per_gpu = site_count + infer_count
    required = len(indices) * cpu_per_gpu
    if len(affinity) < required:
        raise InsufficientCPUAffinityError(
            gpu_count=len(indices),
            site_cpu_per_gpu=site_count,
            infer_cpu_per_gpu=infer_count,
            required_cpu_count=required,
            available_cpu_count=len(affinity),
        )

    gpu_assignments = []
    site_assignments = []
    infer_assignments = []
    for gpu_position, gpu_index in enumerate(indices):
        block_start = gpu_position * cpu_per_gpu
        block = affinity[block_start : block_start + cpu_per_gpu]
        site_cpu_ids = block[:site_count]
        infer_cpu_ids = block[site_count:]
        gpu_assignments.append(
            GPUCPUAssignment(
                gpu_index=gpu_index,
                cpu_ids=block,
                site_cpu_ids=site_cpu_ids,
                infer_cpu_ids=infer_cpu_ids,
            )
        )
        site_assignments.extend(
            CPUWorkerAssignment(
                stage="site",
                gpu_index=gpu_index,
                stage_worker_index=worker_index,
                global_worker_index=gpu_position * site_count + worker_index,
                cpu_id=cpu_id,
            )
            for worker_index, cpu_id in enumerate(site_cpu_ids)
        )
        infer_assignments.extend(
            CPUWorkerAssignment(
                stage="infer",
                gpu_index=gpu_index,
                stage_worker_index=worker_index,
                global_worker_index=gpu_position * infer_count + worker_index,
                cpu_id=cpu_id,
            )
            for worker_index, cpu_id in enumerate(infer_cpu_ids)
        )

    return GPUResourcePlan(
        gpu_indices=indices,
        available_cpu_ids=affinity,
        site_cpu_per_gpu=site_count,
        infer_cpu_per_gpu=infer_count,
        gpu_cpu_assignments=tuple(gpu_assignments),
        site_worker_assignments=tuple(site_assignments),
        infer_worker_assignments=tuple(infer_assignments),
    )


def pin_current_process(cpu_ids: int | Sequence[int]) -> None:
    """Restrict the calling process to one CPU or a resolved GPU CPU block."""

    affinity = (cpu_ids,) if isinstance(cpu_ids, int) else tuple(cpu_ids)
    os.sched_setaffinity(0, set(affinity))


def pin_worker_from_queue(cpu_queue) -> None:
    """Spawn-pool initializer that consumes and binds exactly one CPU ID."""

    pin_current_process(cpu_queue.get())


def gpu_window_atom_budget(gpu_batch_size: int, worker_count: int) -> int:
    return max(4 * gpu_batch_size, 100 * worker_count)
