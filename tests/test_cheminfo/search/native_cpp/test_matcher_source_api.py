"""Compile and execute the Python-independent RI and exact-label orbit contract."""

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


def test_private_matcher_cpp_source_api(tmp_path: Path) -> None:
    repository = Path(__file__).resolve().parents[4]
    native_directory = repository / "hotpot/cheminfo/rules/mathematics/graph_matching/_native"
    graph_directory = repository / "hotpot/cheminfo/rules/mathematics/graph/_native"
    executable = tmp_path / "matcher_source_api"
    command = [
        *_compiler_command(),
        "-std=c++17",
        "-UNDEBUG",
        "-O2",
        "-Wall",
        "-Wextra",
        "-Werror",
        f"-I{native_directory}",
        str(Path(__file__).with_name("matcher_contract.cpp")),
        str(graph_directory / "graph.cpp"),
        str(native_directory / "matcher.cpp"),
        str(native_directory / "automorphism.cpp"),
        "-o",
        str(executable),
    ]

    subprocess.run(command, check=True, capture_output=True, text=True)
    subprocess.run([str(executable)], check=True, capture_output=True, text=True)
