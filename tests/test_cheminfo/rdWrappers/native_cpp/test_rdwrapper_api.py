"""Compile the stable C ABI and header-only C++17 facade."""

from __future__ import annotations

import os
from pathlib import Path
import shlex
import shutil
import subprocess

import pytest


def _compiler(environment_name: str, candidates: tuple[str, ...]) -> list[str]:
    configured = shlex.split(os.environ.get(environment_name, ""))
    if configured and shutil.which(configured[0]):
        return configured
    for candidate in candidates:
        compiler = shutil.which(candidate)
        if compiler:
            return [compiler]
    pytest.skip(f"no {environment_name} compiler is available")


def _include_flags() -> list[str]:
    repository = Path(__file__).resolve().parents[4]
    return [
        f"-I{repository / 'hotpot/cheminfo/_native/include'}",
        f"-I{repository / 'hotpot/cheminfo/rdWrappers/_native/include'}",
    ]


def test_c_api_is_c11_compatible(tmp_path: Path) -> None:
    source = Path(__file__).with_name("c_api_compile.c")
    command = [
        *_compiler("CC", ("cc", "gcc", "clang")),
        "-std=c11",
        "-Wall",
        "-Wextra",
        "-Werror",
        *_include_flags(),
        "-c",
        str(source),
        "-o",
        str(tmp_path / "c_api_compile.o"),
    ]
    subprocess.run(command, check=True, capture_output=True, text=True)


@pytest.mark.parametrize("cxx11_abi", (0, 1))
def test_cpp17_facade_is_abi_neutral(tmp_path: Path, cxx11_abi: int) -> None:
    source = Path(__file__).with_name("rdwrapper_api.cpp")
    executable = tmp_path / f"rdwrapper_api_abi{cxx11_abi}"
    command = [
        *_compiler("CXX", ("c++", "g++", "clang++")),
        "-std=c++17",
        "-UNDEBUG",
        "-O2",
        "-Wall",
        "-Wextra",
        "-Werror",
        f"-D_GLIBCXX_USE_CXX11_ABI={cxx11_abi}",
        *_include_flags(),
        str(source),
        "-o",
        str(executable),
    ]
    subprocess.run(command, check=True, capture_output=True, text=True)
    subprocess.run([str(executable)], check=True, capture_output=True, text=True)
