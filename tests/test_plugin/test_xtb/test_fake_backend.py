"""Self-tests for the deterministic xTB command double."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Protocol

import pytest


class FakeXTBFactory(Protocol):
    def __call__(
        self,
        scenario: str = "success",
        directory_name: str = "fake_xtb",
    ) -> Path: ...


def _write_input(work_directory: Path) -> Path:
    input_path = work_directory / "input.xyz"
    input_path.write_text(
        "2\nfake input\nC 0.0 0.0 0.0\nO 1.42 0.0 0.0\n",
        encoding="utf-8",
    )
    return input_path


@pytest.mark.parametrize(
    "method_arguments, expected_result, absent_result",
    (
        (("--gfn", "2"), "xtbout.json", "gfnff_lists.json"),
        (("--gfnff",), "gfnff_lists.json", "xtbout.json"),
    ),
)
@pytest.mark.parametrize("task_argument", ("--sp", "--opt"))
def test_fake_backend_emits_method_specific_official_artifacts(
    fake_xtb_factory: FakeXTBFactory,
    tmp_path: Path,
    method_arguments: tuple[str, ...],
    expected_result: str,
    absent_result: str,
    task_argument: str,
) -> None:
    executable = fake_xtb_factory()
    work_directory = tmp_path / "direct fake run"
    work_directory.mkdir()
    input_path = _write_input(work_directory)
    arguments = (
        str(executable),
        str(input_path),
        *method_arguments,
        task_argument,
        "--json",
        "--chrg",
        "0",
    )

    completed = subprocess.run(
        arguments,
        cwd=work_directory,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0
    assert completed.stderr == ""
    assert (work_directory / expected_result).is_file()
    assert not (work_directory / absent_result).exists()
    recorded = json.loads(
        (work_directory / "argv.json").read_text(encoding="utf-8")
    )
    assert recorded == list(arguments[1:])
    if task_argument == "--opt":
        assert "GEOMETRY OPTIMIZATION CONVERGED" in completed.stdout
        assert (work_directory / "xtbopt.xyz").is_file()
        assert (work_directory / "xtbopt.log").is_file()
        assert (work_directory / ".xtboptok").is_file()
    else:
        assert not (work_directory / "xtbopt.xyz").exists()
        assert not (work_directory / ".xtboptok").exists()


def test_fake_nonconvergence_omits_the_native_success_marker(
    fake_xtb_factory: FakeXTBFactory,
    tmp_path: Path,
) -> None:
    executable = fake_xtb_factory("nonconvergence")
    work_directory = tmp_path / "nonconverged fake run"
    work_directory.mkdir()
    input_path = _write_input(work_directory)

    completed = subprocess.run(
        (str(executable), str(input_path), "--gfnff", "--opt", "--json"),
        cwd=work_directory,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0
    assert "GEOMETRY OPTIMIZATION DID NOT CONVERGE" in completed.stdout
    assert (work_directory / "xtbopt.xyz").is_file()
    assert not (work_directory / ".xtboptok").exists()


@pytest.mark.parametrize(
    "scenario, return_code, stdout_fragment, stderr_fragment",
    (
        ("success", 0, "xtb version 6.7.1", ""),
        ("version_nonzero", 19, "", "version probe failure"),
        ("version_malformed", 0, "without a parseable version", ""),
    ),
)
def test_fake_backend_version_probe_scenarios_are_deterministic(
    fake_xtb_factory: FakeXTBFactory,
    scenario: str,
    return_code: int,
    stdout_fragment: str,
    stderr_fragment: str,
) -> None:
    executable = fake_xtb_factory(scenario)

    completed = subprocess.run(
        (str(executable), "--version"),
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == return_code
    assert stdout_fragment in completed.stdout
    assert stderr_fragment in completed.stderr
    recorded = json.loads(
        executable.with_name("version_argv.json").read_text(encoding="utf-8")
    )
    assert recorded == ["--version"]
