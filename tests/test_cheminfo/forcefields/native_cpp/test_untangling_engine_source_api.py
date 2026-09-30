"""Build and execute the native ring-untangling C++ API fence."""

from __future__ import annotations

import os
import shlex
import shutil
import subprocess
from importlib.util import find_spec
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


def _openbabel_paths() -> tuple[Path, Path]:
    specification = find_spec("openbabel")
    if specification is None or specification.origin is None:
        pytest.skip("Open Babel is unavailable")
    package = Path(specification.origin).resolve().parent
    include = package / "include/openbabel3"
    library = package / "lib"
    if not include.is_dir() or not library.is_dir():
        pytest.skip("Open Babel native headers or libraries are unavailable")
    return include, library


def test_untangling_engine_cpp_source_api(tmp_path: Path) -> None:
    repository = Path(__file__).resolve().parents[4]
    forcefields = repository / "hotpot/cheminfo/forcefields/_native"
    geometry = repository / "hotpot/cheminfo/geometry/_native"
    graph = repository / "hotpot/cheminfo/graph/_native"
    wrappers = repository / "hotpot/cheminfo/obWrappers/_native"
    include, library = _openbabel_paths()
    executable = tmp_path / "untangling_engine_source_api"
    sources = (
        forcefields / "contracts.cpp",
        forcefields / "session_optimization.cpp",
        forcefields / "structure_session.cpp",
        forcefields / "topology_workspace.cpp",
        forcefields / "untangling_engine.cpp",
        geometry / "batch.cpp",
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
        wrappers / "openbabel_adapter.cpp",
        wrappers / "native_engine.cpp",
        wrappers / "registry.cpp",
        wrappers / "phosphorus_builder.cpp",
        wrappers / "degenerate_torsion.cpp",
    )
    command = [
        *_compiler_command(),
        "-std=c++17",
        "-UNDEBUG",
        "-O2",
        "-Wall",
        "-Wextra",
        "-Werror",
        "-Wno-error=deprecated-copy",
        "-Wno-error=deprecated-declarations",
        "-D_GLIBCXX_USE_CXX11_ABI=0",
        f"-I{forcefields}",
        f"-I{geometry}",
        f"-I{include}",
        str(Path(__file__).with_name("untangling_engine_source_api.cpp")),
        *(str(source) for source in sources),
        f"-L{library}",
        "-lopenbabel",
        "-ldl",
        f"-Wl,-rpath,{library}",
        "-o",
        str(executable),
    ]
    subprocess.run(command, check=True, capture_output=True, text=True)
    subprocess.run([str(executable)], check=True, capture_output=True, text=True)
