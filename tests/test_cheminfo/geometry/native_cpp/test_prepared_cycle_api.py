"""Build and execute the canonical prepared-cycle C++ API fence."""

from __future__ import annotations

import os
from pathlib import Path
import shlex
import shutil
import subprocess

import pytest


def _compiler_command() -> list[str]:
    configured = shlex.split(os.environ.get("CXX", ""))
    if configured and shutil.which(configured[0]):
        return configured
    for candidate in ("c++", "g++", "clang++"):
        compiler = shutil.which(candidate)
        if compiler:
            return [compiler]
    pytest.skip("no C++ compiler is available")


def test_prepared_cycle_cpp_source_api(tmp_path: Path) -> None:
    repository = Path(__file__).resolve().parents[4]
    native_directory = repository / "hotpot/cheminfo/geometry/_native"
    executable = tmp_path / "prepared_cycle_api"
    sources = (
        "cycle_surface.cpp",
        "nonplanar_segment.cpp",
        "nonplanar_surface.cpp",
        "planar_predicates.cpp",
        "prepared_cycle.cpp",
        "primitives.cpp",
        "segment_cycle.cpp",
        "spatial.cpp",
        "triangle_predicates.cpp",
    )
    command = [
        *_compiler_command(),
        "-std=c++17",
        "-UNDEBUG",
        "-O2",
        "-Wall",
        "-Wextra",
        "-Werror",
        f"-I{native_directory}",
        str(Path(__file__).with_name("prepared_cycle_api.cpp")),
        *(str(native_directory / source) for source in sources),
        "-o",
        str(executable),
    ]

    subprocess.run(command, check=True, capture_output=True, text=True)
    subprocess.run([str(executable)], check=True, capture_output=True, text=True)
