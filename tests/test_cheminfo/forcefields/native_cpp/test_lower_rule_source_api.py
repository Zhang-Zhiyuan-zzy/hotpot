"""Run lower ring rules in a source tree without operations or toolkit headers."""

from __future__ import annotations

import os
from pathlib import Path
import re
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


def test_lower_rule_cpp_source_api(tmp_path: Path) -> None:
    repository = Path(__file__).resolve().parents[4]
    fixture = Path(__file__).with_name("lower_rule_source_api.cpp")
    geometry = "hotpot/cheminfo/rules/mathematics/geometry/_native/"
    sources = [
        "hotpot/cheminfo/structure/_native/forcefield_input.cpp",
        "hotpot/cheminfo/rules/physics/forcefields/_native/topology_workspace.cpp",
        *(geometry + name + ".cpp" for name in (
            "batch", "construction", "cycle_surface", "nonplanar_surface",
            "nonplanar_segment", "planar_predicates", "prepared_cycle",
            "primitives", "segment_cycle", "spatial", "triangle_predicates",
        )),
        "hotpot/cheminfo/rules/mathematics/graph/_native/relevant_cycles.cpp",
    ]
    allowed_roots = (
        "hotpot/cheminfo/adapters/native/include/",
        "hotpot/cheminfo/structure/_native/",
        "hotpot/cheminfo/rules/physics/forcefields/_native/",
        "hotpot/cheminfo/reference/_native/",
        geometry,
        "hotpot/cheminfo/rules/mathematics/graph/_native/",
    )
    isolated = tmp_path / "lower-source"
    pending = [repository / name for name in sources]
    pending.extend(repository / name for name in re.findall(
        r'^#include "([^"]+)"', fixture.read_text(), re.MULTILINE
    ))
    copied = set()
    while pending:
        source = pending.pop().resolve()
        if source in copied:
            continue
        relative = source.relative_to(repository)
        assert relative.as_posix().startswith(allowed_roots), str(relative)
        text = source.read_text()
        assert not re.search(r'^\s*#\s*include\s*<openbabel/', text, re.MULTILINE)
        destination = isolated / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
        copied.add(source)
        for include in re.findall(r'^\s*#\s*include\s*"([^"]+)"', text, re.MULTILINE):
            pending.append(source.parent / include)

    executable = tmp_path / "lower_rule_source_api"
    command = [
        *_compiler_command(), "-std=c++17", "-UNDEBUG", "-O2",
        "-Wall", "-Wextra", "-Werror", f"-I{isolated}", str(fixture),
        *(str(isolated / name) for name in sources), "-o", str(executable),
    ]
    subprocess.run(command, check=True, capture_output=True, text=True)
    subprocess.run([str(executable)], check=True, capture_output=True, text=True)
