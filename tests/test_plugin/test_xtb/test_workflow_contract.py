from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from importlib.util import find_spec
from pathlib import Path
from typing import Optional, Protocol

import numpy as np
import pytest

import hotpot
from hotpot.cheminfo.core import Molecule


PUBLIC_XTB_API_AVAILABLE = all(
    find_spec(module_name) is not None
    for module_name in (
        "hotpot.plugins.xtb.contracts",
        "hotpot.plugins.xtb.workflow",
    )
)
if PUBLIC_XTB_API_AVAILABLE:
    from hotpot.plugins.xtb.contracts import (
        GFNXTBMethod,
        XTBExecutionError,
        XTBResultError,
        XTBRunReport,
        XTBTask,
    )
    from hotpot.plugins.xtb.workflow import run_gfn_xtb, run_gfnff


requires_public_xtb_api = pytest.mark.xfail(
    not PUBLIC_XTB_API_AVAILABLE,
    reason="Phase 9 public xTB workflow API is not implemented yet",
    strict=True,
)


class FakeXTBFactory(Protocol):
    def __call__(
        self,
        scenario: str = "success",
        directory_name: str = "fake_xtb",
    ) -> Path: ...


def _molecule() -> Molecule:
    mol = hotpot.read_mol("[H]C([H])([H])O[H]", "smi")
    mol.coordinates = np.asarray(
        (
            (-0.63, 0.90, 0.00),
            (0.00, 0.00, 0.00),
            (-0.63, -0.90, 0.00),
            (0.00, 0.00, 1.00),
            (1.42, 0.00, 0.00),
            (1.82, 0.72, 0.00),
        ),
        dtype=float,
    )
    return mol


def _run(
    workflow: str,
    task: XTBTask,
    mol: Molecule,
    executable: Path,
    work_directory: Optional[Path] = None,
    keep_work_directory: bool = False,
) -> XTBRunReport:
    if workflow == "gfnff":
        return run_gfnff(
            mol,
            task=task,
            charge=0,
            executable=executable,
            work_directory=work_directory,
            keep_work_directory=keep_work_directory,
        )
    return run_gfn_xtb(
        mol,
        method=GFNXTBMethod.GFN2_XTB,
        task=task,
        charge=0,
        unpaired_electrons=0,
        executable=executable,
        work_directory=work_directory,
        keep_work_directory=keep_work_directory,
    )


@requires_public_xtb_api
@pytest.mark.parametrize("workflow", ("gfn2", "gfnff"))
@pytest.mark.parametrize("task_name", ("SINGLEPOINT", "OPTIMIZE"))
def test_gfn_workflows_accept_complete_success_artifacts(
    fake_xtb_factory: FakeXTBFactory,
    workflow: str,
    task_name: str,
) -> None:
    mol = _molecule()
    initial_coordinates = mol.coordinates.copy()
    task = getattr(XTBTask, task_name)

    report = _run(workflow, task, mol, fake_xtb_factory())

    assert report.return_code == 0
    assert report.energy_hartree == pytest.approx(-5.123456789)
    assert report.atom_order_verified is True
    assert report.coordinates_committed is (task is XTBTask.OPTIMIZE)
    assert report.converged is True
    assert report.stderr == ""
    assert Path(report.backend_info.executable).is_absolute()
    assert report.backend_info.version == "6.7.1"
    assert report.argv
    assert report.elapsed_seconds >= 0.0
    expected_result = (
        "gfnff_lists.json" if workflow == "gfnff" else "xtbout.json"
    )
    absent_result = "xtbout.json" if workflow == "gfnff" else "gfnff_lists.json"
    assert expected_result in report.artifacts
    assert absent_result not in report.artifacts
    if task is XTBTask.OPTIMIZE:
        assert ".xtboptok" in report.artifacts
        assert "xtbopt.xyz" in report.artifacts
        assert not np.array_equal(mol.coordinates, initial_coordinates)
    else:
        np.testing.assert_array_equal(mol.coordinates, initial_coordinates)


@requires_public_xtb_api
def test_nonempty_stderr_does_not_turn_success_into_failure(
    fake_xtb_factory: FakeXTBFactory,
) -> None:
    report = _run(
        "gfn2",
        XTBTask.SINGLEPOINT,
        _molecule(),
        fake_xtb_factory("stderr_success"),
    )

    assert report.return_code == 0
    assert report.converged is True
    assert "informational diagnostic" in report.stderr


@requires_public_xtb_api
def test_nonzero_exit_is_execution_failure(
    fake_xtb_factory: FakeXTBFactory,
) -> None:
    mol = _molecule()
    initial_coordinates = mol.coordinates.copy()

    with pytest.raises(XTBExecutionError) as error:
        _run(
            "gfn2",
            XTBTask.SINGLEPOINT,
            mol,
            fake_xtb_factory("nonzero"),
        )

    assert error.value.report.return_code == 17
    assert "fake execution failure" in error.value.report.stderr
    np.testing.assert_array_equal(mol.coordinates, initial_coordinates)


@requires_public_xtb_api
@pytest.mark.parametrize(
    "scenario, task, workflow",
    (
        ("missing_artifact", "SINGLEPOINT", "gfn2"),
        ("nonfinite_energy", "SINGLEPOINT", "gfn2"),
        ("nonfinite_gradient", "SINGLEPOINT", "gfnff"),
        ("nonfinite_coordinates", "OPTIMIZE", "gfn2"),
        ("truncated_geometry", "OPTIMIZE", "gfn2"),
        ("reordered_elements", "OPTIMIZE", "gfn2"),
        ("nonconvergence", "OPTIMIZE", "gfn2"),
    ),
)
def test_invalid_or_incomplete_results_are_rejected_without_coordinate_commit(
    fake_xtb_factory: FakeXTBFactory,
    scenario: str,
    task: str,
    workflow: str,
) -> None:
    mol = _molecule()
    initial_coordinates = mol.coordinates.copy()

    with pytest.raises(XTBResultError) as error:
        _run(
            workflow,
            getattr(XTBTask, task),
            mol,
            fake_xtb_factory(scenario),
        )

    assert error.value.report.coordinates_committed is False
    np.testing.assert_array_equal(mol.coordinates, initial_coordinates)


@requires_public_xtb_api
def test_executable_and_workspace_paths_may_contain_spaces(
    fake_xtb_factory: FakeXTBFactory,
    tmp_path: Path,
) -> None:
    report = _run(
        "gfn2",
        XTBTask.OPTIMIZE,
        _molecule(),
        fake_xtb_factory("success", "fake xtb with spaces"),
        work_directory=tmp_path / "retained workspace with spaces",
        keep_work_directory=True,
    )

    assert report.coordinates_committed is True
    assert " " in str(report.backend_info.executable)
    assert " " in str(report.work_directory)
    assert Path(report.work_directory).is_dir()


@requires_public_xtb_api
def test_simultaneous_runs_use_isolated_workspaces(
    fake_xtb_factory: FakeXTBFactory,
) -> None:
    executable = fake_xtb_factory()

    def run_once(_index: int):
        return _run(
            "gfn2",
            XTBTask.OPTIMIZE,
            _molecule(),
            executable,
            keep_work_directory=True,
        )

    with ThreadPoolExecutor(max_workers=4) as executor:
        reports = tuple(executor.map(run_once, range(8)))

    work_directories = tuple(Path(report.work_directory) for report in reports)
    assert len(set(work_directories)) == len(work_directories)
    assert all(path.is_dir() for path in work_directories)
    assert all(report.coordinates_committed for report in reports)
