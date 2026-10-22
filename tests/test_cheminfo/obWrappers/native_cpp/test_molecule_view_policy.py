"""Compile and execute the Open Babel common-view policy fence."""

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


def test_molecule_view_openbabel_policy_cpp17_api(tmp_path: Path) -> None:
    repository = Path(__file__).resolve().parents[4]
    wrappers = repository / "hotpot/cheminfo/obWrappers/_native"
    common = repository / "hotpot/cheminfo/_native/include"
    include, library = _openbabel_paths()
    executable = tmp_path / "molecule_view_policy"
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
        f"-I{wrappers}",
        f"-I{common}",
        "-isystem",
        str(include),
        str(Path(__file__).with_name("molecule_view_policy.cpp")),
        str(wrappers / "molecule_data.cpp"),
        str(wrappers / "openbabel_adapter.cpp"),
        str(wrappers / "registry.cpp"),
        f"-L{library}",
        "-lopenbabel",
        f"-Wl,-rpath,{library}",
        "-o",
        str(executable),
    ]

    subprocess.run(command, check=True, capture_output=True, text=True)
    subprocess.run([str(executable)], check=True, capture_output=True, text=True)
