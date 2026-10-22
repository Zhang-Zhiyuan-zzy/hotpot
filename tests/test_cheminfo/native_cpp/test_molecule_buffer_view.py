"""Build and execute the consumer-neutral C++17 molecule-view fence."""

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


def test_molecule_buffer_view_cpp17_api(tmp_path: Path) -> None:
    repository = Path(__file__).resolve().parents[3]
    include_directory = repository / "hotpot/cheminfo/_native/include"
    executable = tmp_path / "molecule_buffer_view"
    command = [
        *_compiler_command(),
        "-std=c++17",
        "-UNDEBUG",
        "-O2",
        "-Wall",
        "-Wextra",
        "-Werror",
        f"-I{include_directory}",
        str(Path(__file__).with_name("molecule_buffer_view.cpp")),
        "-o",
        str(executable),
    ]

    subprocess.run(command, check=True, capture_output=True, text=True)
    subprocess.run([str(executable)], check=True, capture_output=True, text=True)
