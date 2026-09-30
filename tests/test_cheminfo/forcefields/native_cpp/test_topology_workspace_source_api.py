"""Build and execute the shared topology-workspace C++ source API fence."""

from __future__ import annotations

import os
import shlex
import shutil
import subprocess
from pathlib import Path

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


def test_topology_workspace_cpp_source_api(tmp_path: Path) -> None:
    repository = Path(__file__).resolve().parents[4]
    forcefields = repository / "hotpot/cheminfo/forcefields/_native"
    geometry = repository / "hotpot/cheminfo/geometry/_native"
    graph = repository / "hotpot/cheminfo/graph/_native"
    wrappers = repository / "hotpot/cheminfo/obWrappers/_native"
    executable = tmp_path / "topology_workspace_source_api"
    sources = (
        forcefields / "contracts.cpp",
        forcefields / "topology_workspace.cpp",
        geometry / "batch.cpp",
        geometry / "construction.cpp",
        geometry / "cycle_surface.cpp",
        geometry / "nonplanar_surface.cpp",
        geometry / "nonplanar_segment.cpp",
        geometry / "planar_predicates.cpp",
        geometry / "prepared_cycle.cpp",
        geometry / "primitives.cpp",
        geometry / "segment_cycle.cpp",
        geometry / "spatial.cpp",
        geometry / "triangle_predicates.cpp",
        graph / "relevant_cycles.cpp",
        wrappers / "molecule_data.cpp",
    )
    command = [
        *_compiler_command(),
        "-std=c++17",
        "-UNDEBUG",
        "-O2",
        "-Wall",
        "-Wextra",
        "-Werror",
        f"-I{forcefields}",
        f"-I{geometry}",
        str(Path(__file__).with_name("topology_workspace_source_api.cpp")),
        *(str(source) for source in sources),
        "-o",
        str(executable),
    ]
    subprocess.run(command, check=True, capture_output=True, text=True)
    subprocess.run([str(executable)], check=True, capture_output=True, text=True)
