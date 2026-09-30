"""Compile the native workflow overloads exposed to C++ callers."""

from __future__ import annotations

from importlib.util import find_spec
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


def _openbabel_include() -> Path:
    specification = find_spec("openbabel")
    if specification is None or specification.origin is None:
        pytest.skip("Open Babel is unavailable")
    include = Path(specification.origin).resolve().parent / "include/openbabel3"
    if not include.is_dir():
        pytest.skip("Open Babel native headers are unavailable")
    return include


def test_workflow_stage_cpp_header_api(tmp_path: Path) -> None:
    repository = Path(__file__).resolve().parents[4]
    forcefields = repository / "hotpot/cheminfo/forcefields/_native"
    include = _openbabel_include()
    output = tmp_path / "workflow_stage_header_api.o"
    command = [
        *_compiler_command(),
        "-std=c++17",
        "-Wall",
        "-Wextra",
        "-Werror",
        f"-I{forcefields}",
        "-isystem",
        str(include),
        "-c",
        str(Path(__file__).with_name("workflow_stage_header_api.cpp")),
        "-o",
        str(output),
    ]

    subprocess.run(command, check=True, capture_output=True, text=True)
