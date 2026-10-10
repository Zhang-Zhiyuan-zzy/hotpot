"""Executable-resolution contracts for the external-process harness."""

from __future__ import annotations

import os
from pathlib import Path

def _write_executable(directory: Path, name: str) -> Path:
    executable = directory / name
    executable.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    executable.chmod(executable.stat().st_mode | 0o111)
    return executable


def test_explicit_executable_precedes_environment_and_path(tmp_path: Path) -> None:
    from hotpot.plugins._harness.executable import resolve_executable

    explicit_dir = tmp_path / "explicit"
    environment_dir = tmp_path / "environment"
    path_dir = tmp_path / "path"
    for directory in (explicit_dir, environment_dir, path_dir):
        directory.mkdir()

    explicit = _write_executable(explicit_dir, "probe")
    environment = _write_executable(environment_dir, "probe")
    _write_executable(path_dir, "probe")

    resolved = resolve_executable(
        "probe",
        explicit=explicit,
        env_var="HOTPOT_TEST_EXECUTABLE",
        env={
            "HOTPOT_TEST_EXECUTABLE": os.fspath(environment),
            "PATH": os.fspath(path_dir),
        },
    )

    assert resolved == explicit.resolve()
    assert resolved.is_absolute()


def test_environment_executable_precedes_path(tmp_path: Path) -> None:
    from hotpot.plugins._harness.executable import resolve_executable

    environment_dir = tmp_path / "environment"
    path_dir = tmp_path / "path"
    environment_dir.mkdir()
    path_dir.mkdir()
    environment = _write_executable(environment_dir, "probe")
    _write_executable(path_dir, "probe")

    resolved = resolve_executable(
        "probe",
        env_var="HOTPOT_TEST_EXECUTABLE",
        env={
            "HOTPOT_TEST_EXECUTABLE": os.fspath(environment),
            "PATH": os.fspath(path_dir),
        },
    )

    assert resolved == environment.resolve()
    assert resolved.is_absolute()


def test_path_executable_is_returned_as_an_absolute_path(tmp_path: Path) -> None:
    from hotpot.plugins._harness.executable import resolve_executable

    path_dir = tmp_path / "path"
    path_dir.mkdir()
    executable = _write_executable(path_dir, "probe")

    resolved = resolve_executable(
        "probe",
        env_var="HOTPOT_TEST_EXECUTABLE",
        env={"PATH": os.fspath(path_dir)},
    )

    assert resolved == executable.resolve()
    assert resolved.is_absolute()
