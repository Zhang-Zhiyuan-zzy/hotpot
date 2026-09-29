"""Profile the public geometry boundary before and after native migration."""

from __future__ import annotations

import argparse
import cProfile
import importlib.util
import json
import platform
import pstats
import statistics
import time
import tracemalloc
from pathlib import Path
from typing import Callable, Dict, Sequence

import numpy as np

from hotpot.cheminfo import geometry
from tests.geometry_characterization import (
    JSONValue,
    build_geometry_characterization,
)


_NATIVE_MODULE = "hotpot.cheminfo.geometry._geometry_native"


def _percentile(samples: Sequence[float], fraction: float) -> float:
    ordered = sorted(samples)
    position = fraction * (len(ordered) - 1)
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _profile_calls(function: Callable[[], object]) -> Dict[str, JSONValue]:
    profiler = cProfile.Profile()
    profiler.enable()
    function()
    profiler.disable()

    statistics_by_function = pstats.Stats(profiler).stats
    rows = [
        {
            "file": filename,
            "line": line,
            "function": function_name,
            "primitive_calls": primitive_calls,
            "total_calls": total_calls,
            "self_seconds": self_seconds,
            "cumulative_seconds": cumulative_seconds,
        }
        for (
            filename,
            line,
            function_name,
        ), (
            primitive_calls,
            total_calls,
            self_seconds,
            cumulative_seconds,
            _callers,
        ) in statistics_by_function.items()
    ]
    rows.sort(key=lambda row: row["cumulative_seconds"], reverse=True)

    return {
        "total_profiled_calls": sum(row["total_calls"] for row in rows),
        "python_geometry_calls": sum(
            row["total_calls"]
            for row in rows
            if "/hotpot/cheminfo/geometry/" in row["file"]
            and row["file"].endswith(".py")
        ),
        "native_geometry_calls": sum(
            row["total_calls"]
            for row in rows
            if "_geometry_native" in row["file"]
            or "_geometry_native" in row["function"]
        ),
        "top_cumulative_functions": rows[:20],
    }


def run_profile(repeats: int) -> Dict[str, JSONValue]:
    specification = importlib.util.find_spec(_NATIVE_MODULE)
    timings = []
    tracemalloc.start()
    for _ in range(repeats):
        started = time.perf_counter_ns()
        build_geometry_characterization()
        timings.append((time.perf_counter_ns() - started) / 1.0e6)
    _, peak_bytes = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    callable_modules = {
        name: getattr(getattr(geometry, name), "__module__", None)
        for name in geometry.__all__
        if callable(getattr(geometry, name))
    }
    return {
        "schema_version": 1,
        "runtime": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "numpy": np.__version__,
        },
        "native_geometry_module": {
            "name": _NATIVE_MODULE,
            "available": specification is not None,
            "origin": None if specification is None else specification.origin,
        },
        "public_callable_modules": callable_modules,
        "workload": {
            "name": "geometry_native_migration_characterization",
            "repeats": repeats,
            "median_ms": statistics.median(timings),
            "p95_ms": _percentile(timings, 0.95),
            "minimum_ms": min(timings),
            "maximum_ms": max(timings),
            "peak_traced_bytes": peak_bytes,
        },
        "profile": _profile_calls(build_geometry_characterization),
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Profile the public geometry implementation boundary.",
    )
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args()

    report = run_profile(arguments.repeats)
    rendered = json.dumps(report, indent=2, sort_keys=True)
    if arguments.output is not None:
        arguments.output.parent.mkdir(parents=True, exist_ok=True)
        arguments.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
