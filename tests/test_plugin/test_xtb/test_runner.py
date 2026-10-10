"""Typed argv, execution, and artifact contracts for the xTB runner."""

from __future__ import annotations

import json
import os
from importlib.util import find_spec
from pathlib import Path
from typing import Optional, Protocol

import pytest


PHASE10_RUNNER_API_AVAILABLE = all(
    find_spec(module_name) is not None
    for module_name in (
        "hotpot.plugins.xtb.backend",
        "hotpot.plugins.xtb.contracts",
        "hotpot.plugins.xtb.runner",
    )
)
if PHASE10_RUNNER_API_AVAILABLE:
    from hotpot.plugins.xtb.backend import probe_xtb_backend
    from hotpot.plugins.xtb.contracts import (
        XTBExecutionError,
        XTBInputError,
        XTBMethod,
        XTBRequest,
        XTBResultError,
        XTBTask,
    )
    from hotpot.plugins.xtb.runner import run_xtb


requires_phase10_runner = pytest.mark.xfail(
    not PHASE10_RUNNER_API_AVAILABLE,
    reason="Phase 10 typed xTB runner is not implemented yet",
    strict=True,
)


class FakeXTBFactory(Protocol):
    def __call__(
        self,
        scenario: str = "success",
        directory_name: str = "fake_xtb",
    ) -> Path: ...


def _write_input(work_directory: Path) -> Path:
    work_directory.mkdir()
    input_path = work_directory / "input.xyz"
    input_path.write_text(
        "2\nfake input\nC 0.0 0.0 0.0\nO 1.42 0.0 0.0\n",
        encoding="utf-8",
    )
    return input_path


def _request(
    executable: Path,
    work_directory: Path,
    *,
    method_name: str,
    task_name: str,
    charge: int = -1,
    unpaired_electrons: Optional[int] = None,
) -> XTBRequest:
    backend_info = probe_xtb_backend(
        executable=executable,
        environment=dict(os.environ),
    )
    return XTBRequest(
        backend_info=backend_info,
        method=getattr(XTBMethod, method_name),
        task=getattr(XTBTask, task_name),
        input_path=_write_input(work_directory),
        work_directory=work_directory,
        charge=charge,
        unpaired_electrons=unpaired_electrons,
        environment=dict(os.environ),
        timeout_seconds=None,
    )


@requires_phase10_runner
@pytest.mark.parametrize(
    "method_name, task_name, expected_method, expected_task, result_name",
    (
        ("GFNFF", "SINGLEPOINT", ("--gfnff",), "--sp", "gfnff_lists.json"),
        ("GFNFF", "OPTIMIZE", ("--gfnff",), "--opt", "gfnff_lists.json"),
        ("GFN2_XTB", "SINGLEPOINT", ("--gfn", "2"), "--sp", "xtbout.json"),
        ("GFN2_XTB", "OPTIMIZE", ("--gfn", "2"), "--opt", "xtbout.json"),
    ),
)
def test_runner_builds_method_and_task_specific_argv_and_artifacts(
    fake_xtb_factory: FakeXTBFactory,
    tmp_path: Path,
    method_name: str,
    task_name: str,
    expected_method: tuple[str, ...],
    expected_task: str,
    result_name: str,
) -> None:
    executable = fake_xtb_factory()
    request = _request(
        executable,
        tmp_path / "run",
        method_name=method_name,
        task_name=task_name,
        unpaired_electrons=2 if method_name == "GFN2_XTB" else None,
    )
    executable.with_name("version_argv.json").unlink()

    report = run_xtb(request)

    recorded = json.loads(
        (request.work_directory / "argv.json").read_text(encoding="utf-8")
    )
    assert tuple(recorded) == report.argv[1:]
    assert all(item in report.argv for item in expected_method)
    assert expected_task in report.argv
    assert "--chrg" in report.argv
    assert report.argv[report.argv.index("--chrg") + 1] == "-1"
    assert result_name in report.artifacts
    assert not executable.with_name("version_argv.json").exists()
    if method_name == "GFN2_XTB":
        assert report.argv[report.argv.index("--uhf") + 1] == "2"
    else:
        assert "--uhf" not in report.argv
    if task_name == "OPTIMIZE":
        assert ".xtboptok" in report.artifacts


@requires_phase10_runner
def test_gfnff_rejects_unpaired_electrons_instead_of_forwarding_uhf(
    fake_xtb_factory: FakeXTBFactory,
    tmp_path: Path,
) -> None:
    request = _request(
        fake_xtb_factory(),
        tmp_path / "run",
        method_name="GFNFF",
        task_name="SINGLEPOINT",
        unpaired_electrons=2,
    )

    with pytest.raises(XTBInputError):
        run_xtb(request)


@requires_phase10_runner
def test_stderr_text_is_preserved_without_turning_success_into_failure(
    fake_xtb_factory: FakeXTBFactory,
    tmp_path: Path,
) -> None:
    request = _request(
        fake_xtb_factory("stderr_success"),
        tmp_path / "run",
        method_name="GFN2_XTB",
        task_name="SINGLEPOINT",
        unpaired_electrons=0,
    )

    report = run_xtb(request)

    assert report.return_code == 0
    assert "informational diagnostic" in report.stderr


@requires_phase10_runner
def test_nonzero_exit_raises_with_complete_failure_report(
    fake_xtb_factory: FakeXTBFactory,
    tmp_path: Path,
) -> None:
    request = _request(
        fake_xtb_factory("nonzero"),
        tmp_path / "run",
        method_name="GFN2_XTB",
        task_name="SINGLEPOINT",
        unpaired_electrons=0,
    )

    with pytest.raises(XTBExecutionError) as error:
        run_xtb(request)

    assert error.value.report.return_code == 17
    assert "fake execution failure" in error.value.report.stderr


@requires_phase10_runner
def test_missing_method_specific_result_is_an_explicit_result_failure(
    fake_xtb_factory: FakeXTBFactory,
    tmp_path: Path,
) -> None:
    request = _request(
        fake_xtb_factory("missing_artifact"),
        tmp_path / "run",
        method_name="GFNFF",
        task_name="SINGLEPOINT",
    )

    with pytest.raises(XTBResultError) as error:
        run_xtb(request)

    assert error.value.report.return_code == 0
    assert "gfnff_lists.json" not in error.value.report.artifacts
