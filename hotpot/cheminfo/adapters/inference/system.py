"""Host CPU-affinity observations and process/pool binding."""
from __future__ import annotations
import os
from typing import Sequence


def current_cpu_affinity() -> tuple[int, ...]:
    """Return the CPUs made available by the enclosing cpuset/cgroup."""

    return tuple(sorted(os.sched_getaffinity(0)))


def pin_current_process(cpu_ids: int | Sequence[int]) -> None:
    """Restrict the calling process to one CPU or a resolved GPU CPU block."""

    affinity = (cpu_ids,) if isinstance(cpu_ids, int) else tuple(cpu_ids)
    os.sched_setaffinity(0, set(affinity))


def pin_worker_from_queue(cpu_queue) -> None:
    """Spawn-pool initializer that consumes and binds exactly one CPU ID."""

    pin_current_process(cpu_queue.get())
