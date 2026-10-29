"""Validated scientific-state projection use in xTB workflows."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping, Optional, Protocol, Union

import numpy as np
import pytest

import hotpot
from hotpot.cheminfo.calculator.execution_selection import BackendStateProjection
from hotpot.cheminfo.core import Molecule
from hotpot.plugins.xtb import workflow as xtb_workflow
from hotpot.plugins.xtb.contracts import (
    GFNXTBMethod,
    XTBApplicabilityError,
    XTBBackendInfo,
    XTBExecutionError,
    XTBInputError,
    XTBMethod,
    XTBTask,
)


class FakeXTBFactory(Protocol):
    def __call__(
        self,
        scenario: str = "success",
        directory_name: str = "fake_xtb",
    ) -> Path: ...


class XTBProjectionFactory(Protocol):
    def __call__(
        self,
        mol: Molecule,
        executable: Path,
        method: XTBMethod,
        charge: int = 0,
        uhf: Optional[int] = None,
    ) -> BackendStateProjection: ...


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


def test_gfn_xtb_consumes_validated_projection_without_reinference(
    fake_xtb_factory: FakeXTBFactory,
    xtb_projection_factory: XTBProjectionFactory,
) -> None:
    mol = _molecule()
    executable = fake_xtb_factory()
    state = xtb_projection_factory(
        mol,
        executable,
        XTBMethod.GFN2_XTB,
        0,
        0,
    )

    report = xtb_workflow.run_gfn_xtb(
        mol,
        state=state,
        task=XTBTask.SINGLEPOINT,
        executable=executable,
    )

    assert report.charge == state.charge
    assert report.unpaired_electrons == state.uhf
    assert report.scientific_state_identity == state.scientific_state_identity
    assert report.facts_identity == state.facts_identity
    assert report.projection_convention == state.projection_convention


def test_projection_for_another_method_is_rejected(
    fake_xtb_factory: FakeXTBFactory,
    xtb_projection_factory: XTBProjectionFactory,
) -> None:
    mol = _molecule()
    executable = fake_xtb_factory()
    gfnff_state = xtb_projection_factory(
        mol,
        executable,
        XTBMethod.GFNFF,
    )

    with pytest.raises(XTBInputError, match="does not match"):
        xtb_workflow.run_gfn_xtb(
            mol,
            state=gfnff_state,
            method=GFNXTBMethod.GFN2_XTB,
            task=XTBTask.SINGLEPOINT,
            executable=executable,
        )


def test_projection_for_another_molecule_is_rejected(
    fake_xtb_factory: FakeXTBFactory,
    xtb_projection_factory: XTBProjectionFactory,
) -> None:
    mol = _molecule()
    executable = fake_xtb_factory()
    state = xtb_projection_factory(
        mol,
        executable,
        XTBMethod.GFN2_XTB,
        0,
        0,
    )
    other = hotpot.read_mol("[H]O[H]", "smi")
    other.coordinates = np.asarray(
        ((-0.8, 0.0, 0.0), (0.0, 0.0, 0.0), (0.8, 0.0, 0.0)),
        dtype=float,
    )

    with pytest.raises(XTBInputError, match="molecular facts"):
        xtb_workflow.run_gfn_xtb(
            other,
            state=state,
            task=XTBTask.SINGLEPOINT,
            executable=executable,
        )


def test_geometry_is_validated_before_backend_probe(
    fake_xtb_factory: FakeXTBFactory,
    xtb_projection_factory: XTBProjectionFactory,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mol = _molecule()
    executable = fake_xtb_factory()
    state = xtb_projection_factory(mol, executable, XTBMethod.GFNFF)
    coordinates = mol.coordinates.copy()
    coordinates[0, 0] = np.nan
    mol.coordinates = coordinates

    def reject_probe(
        executable: Optional[Union[str, Path]] = None,
        *,
        environment: Optional[Mapping[str, str]] = None,
        timeout_seconds: Optional[float] = 30.0,
    ) -> XTBBackendInfo:
        raise AssertionError("Backend probe must follow geometry validation")

    monkeypatch.setattr(xtb_workflow, "probe_xtb_backend", reject_probe)

    with pytest.raises(XTBInputError, match="non-finite"):
        xtb_workflow.run_gfnff(
            mol,
            state=state,
            executable=executable,
        )


def test_each_workflow_execution_probes_backend_once(
    fake_xtb_factory: FakeXTBFactory,
    xtb_projection_factory: XTBProjectionFactory,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mol = _molecule()
    executable = fake_xtb_factory()
    state = xtb_projection_factory(
        mol,
        executable,
        XTBMethod.GFN2_XTB,
        0,
        0,
    )
    probe_count = 0
    probe = xtb_workflow.probe_xtb_backend

    def count_probe(
        executable: Optional[Union[str, Path]] = None,
        *,
        environment: Optional[Mapping[str, str]] = None,
        timeout_seconds: Optional[float] = 30.0,
    ) -> XTBBackendInfo:
        nonlocal probe_count
        probe_count += 1
        return probe(
            executable,
            environment=environment,
            timeout_seconds=timeout_seconds,
        )

    monkeypatch.setattr(xtb_workflow, "probe_xtb_backend", count_probe)

    xtb_workflow.run_gfn_xtb(
        mol,
        state=state,
        task=XTBTask.SINGLEPOINT,
        executable=executable,
    )

    assert probe_count == 1


def test_retained_workspace_is_a_unique_child_of_requested_parent(
    fake_xtb_factory: FakeXTBFactory,
    xtb_projection_factory: XTBProjectionFactory,
    tmp_path: Path,
) -> None:
    mol = _molecule()
    executable = fake_xtb_factory()
    state = xtb_projection_factory(mol, executable, XTBMethod.GFNFF)
    workspace_parent = tmp_path / "retained parent"

    first = xtb_workflow.run_gfnff(
        mol,
        state=state,
        task=XTBTask.SINGLEPOINT,
        executable=executable,
        work_directory=workspace_parent,
        keep_work_directory=True,
    )
    second = xtb_workflow.run_gfnff(
        mol,
        state=state,
        task=XTBTask.SINGLEPOINT,
        executable=executable,
        work_directory=workspace_parent,
        keep_work_directory=True,
    )

    assert first.work_directory.parent == workspace_parent.resolve()
    assert second.work_directory.parent == workspace_parent.resolve()
    assert first.work_directory != second.work_directory
    assert first.workspace_retained is True
    assert second.workspace_retained is True
    assert first.work_directory.is_dir()
    assert second.work_directory.is_dir()


def test_gfnff_am_input_is_rejected_before_numerical_run(
    fake_xtb_factory: FakeXTBFactory,
    xtb_projection_factory: XTBProjectionFactory,
    tmp_path: Path,
) -> None:
    executable = fake_xtb_factory()
    state = xtb_projection_factory(_molecule(), executable, XTBMethod.GFNFF)
    mol = hotpot.read_mol("[Am+3]", "smi")
    mol.coordinates = np.zeros((1, 3), dtype=float)
    workspace_parent = tmp_path / "am-preflight"

    with pytest.raises(XTBApplicabilityError, match="unsupported input"):
        xtb_workflow.run_gfnff(
            mol,
            state=state,
            task=XTBTask.SINGLEPOINT,
            executable=executable,
            work_directory=workspace_parent,
            keep_work_directory=True,
        )

    workspaces = tuple(workspace_parent.iterdir())
    assert len(workspaces) == 1
    assert (workspaces[0] / "input.sdf").is_file()
    assert not (workspaces[0] / "argv.json").exists()


def test_default_workspace_is_removed_after_success(
    fake_xtb_factory: FakeXTBFactory,
    xtb_projection_factory: XTBProjectionFactory,
    tmp_path: Path,
) -> None:
    mol = _molecule()
    executable = fake_xtb_factory()
    state = xtb_projection_factory(
        mol,
        executable,
        XTBMethod.GFN2_XTB,
        0,
        0,
    )
    workspace_parent = tmp_path / "workflow parent"

    report = xtb_workflow.run_gfn_xtb(
        mol,
        state=state,
        task=XTBTask.SINGLEPOINT,
        executable=executable,
        work_directory=workspace_parent,
    )

    assert report.work_directory.parent == workspace_parent.resolve()
    assert report.workspace_retained is False
    assert not report.work_directory.exists()
    assert tuple(workspace_parent.iterdir()) == ()


def test_failed_run_removes_workspace_unless_retention_is_requested(
    fake_xtb_factory: FakeXTBFactory,
    xtb_projection_factory: XTBProjectionFactory,
    tmp_path: Path,
) -> None:
    mol = _molecule()
    executable = fake_xtb_factory("nonzero")
    state = xtb_projection_factory(
        mol,
        executable,
        XTBMethod.GFN2_XTB,
        0,
        0,
    )
    workspace_parent = tmp_path / "failed-run-parent"

    with pytest.raises(XTBExecutionError) as error:
        xtb_workflow.run_gfn_xtb(
            mol,
            state=state,
            task=XTBTask.SINGLEPOINT,
            executable=executable,
            work_directory=workspace_parent,
        )

    assert error.value.report.workspace_retained is False
    assert not error.value.report.work_directory.exists()
    assert tuple(workspace_parent.iterdir()) == ()


def test_optimization_rejects_source_changes_during_external_execution(
    fake_xtb_factory: FakeXTBFactory,
    xtb_projection_factory: XTBProjectionFactory,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mol = _molecule()
    executable = fake_xtb_factory()
    state = xtb_projection_factory(mol, executable, XTBMethod.GFN2_XTB, 0, 0)
    changed = mol.coordinates + 7
    execute = xtb_workflow.run_xtb

    def mutate_source(request):
        report = execute(request)
        mol.coordinates = changed
        return report

    monkeypatch.setattr(xtb_workflow, "run_xtb", mutate_source)
    with pytest.raises(RuntimeError, match="geometry.*changed"):
        xtb_workflow.run_gfn_xtb(
            mol, state=state, task=XTBTask.OPTIMIZE, executable=executable,
        )
    np.testing.assert_array_equal(mol.coordinates, changed)
