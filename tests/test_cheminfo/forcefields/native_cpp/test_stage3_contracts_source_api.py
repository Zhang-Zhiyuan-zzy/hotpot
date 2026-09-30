"""Build and execute the Stage-3 force-field contract C++ API fence."""

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


def test_stage3_contracts_cpp_source_api(tmp_path: Path) -> None:
    repository = Path(__file__).resolve().parents[4]
    forcefields = repository / "hotpot/cheminfo/forcefields/_native"
    executable = tmp_path / "stage3_contracts_source_api"
    command = [
        *_compiler_command(),
        "-std=c++17",
        "-UNDEBUG",
        "-O2",
        "-Wall",
        "-Wextra",
        "-Werror",
        f"-I{forcefields}",
        str(Path(__file__).with_name("stage3_contracts_source_api.cpp")),
        str(forcefields / "stage_contracts.cpp"),
        str(forcefields / "trajectory.cpp"),
        "-o",
        str(executable),
    ]
    subprocess.run(command, check=True, capture_output=True, text=True)
    subprocess.run([str(executable)], check=True, capture_output=True, text=True)
