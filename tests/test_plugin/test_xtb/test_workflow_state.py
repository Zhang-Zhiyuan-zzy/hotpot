from __future__ import annotations

from pathlib import Path
from typing import Mapping, Optional, Protocol, Union

import numpy as np
import pytest

import hotpot
from hotpot.cheminfo.calculator import infer_charge
from hotpot.cheminfo.calculator.electronic_state import (
    ChargeInferenceResult,
    ChargeInferenceSource,
    ElectronicState,
    SpinInferenceSource,
)
from hotpot.cheminfo.core import Molecule
from hotpot.plugins.xtb import workflow as xtb_workflow
from hotpot.plugins.xtb.contracts import (
    XTBApplicabilityError,
    XTBBackendInfo,
    XTBExecutionError,
    XTBInputError,
    XTBTask,
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


def _explicit_state() -> ElectronicState:
    return ElectronicState(
        charge=0,
        unpaired_electrons=0,
        multiplicity=1,
        fragment_charges=(0,),
        charge_source=ChargeInferenceSource.PRESERVED,
        spin_source=SpinInferenceSource.EXPLICIT,
        assumptions=("Test state is authoritative.",),
    )


def test_gfn_xtb_consumes_existing_state_without_reinference(
    fake_xtb_factory: FakeXTBFactory,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def reject_resolution(
        mol: Molecule,
        *,
        charge: Optional[int] = None,
        unpaired_electrons: Optional[int] = None,
    ) -> ElectronicState:
        raise AssertionError("Existing electronic state must be consumed directly")

    monkeypatch.setattr(xtb_workflow, "resolve_electronic_state", reject_resolution)
    state = _explicit_state()

    report = xtb_workflow.run_gfn_xtb(
        _molecule(),
        state=state,
        task=XTBTask.SINGLEPOINT,
        executable=fake_xtb_factory(),
    )

    assert report.charge == state.charge
    assert report.unpaired_electrons == state.unpaired_electrons
    assert report.charge_source is state.charge_source
    assert report.spin_source is state.spin_source
    assert report.fragment_charges == state.fragment_charges
    assert report.state_assumptions == state.assumptions


@pytest.mark.parametrize(
    "overrides",
    ({"charge": 0}, {"unpaired_electrons": 0}),
)
def test_existing_state_cannot_be_mixed_with_explicit_overrides(
    overrides: Mapping[str, int],
) -> None:
    with pytest.raises(XTBInputError, match="cannot be mixed"):
        xtb_workflow.run_gfn_xtb(
            _molecule(),
            state=_explicit_state(),
            **overrides,
        )


def test_gfnff_charge_state_cannot_be_mixed_with_explicit_charge() -> None:
    mol = _molecule()

    with pytest.raises(XTBInputError, match="cannot be supplied together"):
        xtb_workflow.run_gfnff(
            mol,
            charge_state=infer_charge(mol),
            charge=0,
        )


def test_gfnff_consumes_existing_charge_state_without_reinference(
    fake_xtb_factory: FakeXTBFactory,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mol = _molecule()
    charge_state = infer_charge(mol)

    def reject_charge_inference(mol: Molecule) -> ChargeInferenceResult:
        raise AssertionError("Existing charge state must be consumed directly")

    monkeypatch.setattr(xtb_workflow, "infer_charge", reject_charge_inference)

    report = xtb_workflow.run_gfnff(
        mol,
        task=XTBTask.SINGLEPOINT,
        charge_state=charge_state,
        executable=fake_xtb_factory(),
    )

    assert report.charge == charge_state.total_charge
    assert report.charge_source is charge_state.source
    assert report.fragment_charges == tuple(
        fragment.charge for fragment in charge_state.fragments
    )
    assert report.state_assumptions == charge_state.assumptions
    assert report.spin_source is None


def test_default_gfn_xtb_state_reports_calculator_provenance(
    fake_xtb_factory: FakeXTBFactory,
) -> None:
    report = xtb_workflow.run_gfn_xtb(
        _molecule(),
        task=XTBTask.SINGLEPOINT,
        executable=fake_xtb_factory(),
    )

    assert report.charge == 0
    assert report.unpaired_electrons == 0
    assert report.charge_source is ChargeInferenceSource.VALENCE
    assert report.spin_source is SpinInferenceSource.LOWEST_SPIN_PARITY
    assert report.fragment_charges == (0,)
    assert report.state_assumptions


def test_gfnff_does_not_resolve_or_infer_spin(
    fake_xtb_factory: FakeXTBFactory,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def reject_resolution(
        mol: Molecule,
        *,
        charge: Optional[int] = None,
        unpaired_electrons: Optional[int] = None,
    ) -> ElectronicState:
        raise AssertionError("GFN-FF must not resolve a spin state")

    monkeypatch.setattr(xtb_workflow, "resolve_electronic_state", reject_resolution)

    report = xtb_workflow.run_gfnff(
        _molecule(),
        task=XTBTask.SINGLEPOINT,
        executable=fake_xtb_factory(),
    )

    assert report.unpaired_electrons is None
    assert report.spin_source is None
    assert report.charge_source is ChargeInferenceSource.VALENCE


def test_geometry_is_validated_before_backend_probe(
    fake_xtb_factory: FakeXTBFactory,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mol = _molecule()
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
            charge=0,
            executable=fake_xtb_factory(),
        )


def test_each_workflow_call_probes_backend_once(
    fake_xtb_factory: FakeXTBFactory,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
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
        _molecule(),
        task=XTBTask.SINGLEPOINT,
        charge=0,
        unpaired_electrons=0,
        executable=fake_xtb_factory(),
    )

    assert probe_count == 1


def test_gfnff_am_input_is_rejected_before_numerical_run(
    fake_xtb_factory: FakeXTBFactory,
    tmp_path: Path,
) -> None:
    mol = hotpot.read_mol("[Am+3]", "smi")
    mol.coordinates = np.zeros((1, 3), dtype=float)
    workspace_parent = tmp_path / "am-preflight"

    with pytest.raises(XTBApplicabilityError, match="unsupported input"):
        xtb_workflow.run_gfnff(
            mol,
            task=XTBTask.SINGLEPOINT,
            charge=3,
            executable=fake_xtb_factory(),
            work_directory=workspace_parent,
            keep_work_directory=True,
        )

    workspaces = tuple(workspace_parent.iterdir())
    assert len(workspaces) == 1
    assert (workspaces[0] / "input.xyz").is_file()
    assert not (workspaces[0] / "argv.json").exists()


def test_work_directory_is_parent_and_default_workspace_is_removed(
    fake_xtb_factory: FakeXTBFactory,
    tmp_path: Path,
) -> None:
    workspace_parent = tmp_path / "workflow parent"

    report = xtb_workflow.run_gfn_xtb(
        _molecule(),
        task=XTBTask.SINGLEPOINT,
        charge=0,
        unpaired_electrons=0,
        executable=fake_xtb_factory(),
        work_directory=workspace_parent,
    )

    assert report.work_directory.parent == workspace_parent.resolve()
    assert report.workspace_retained is False
    assert not report.work_directory.exists()
    assert tuple(workspace_parent.iterdir()) == ()


def test_retained_workspace_is_a_unique_child_of_requested_parent(
    fake_xtb_factory: FakeXTBFactory,
    tmp_path: Path,
) -> None:
    workspace_parent = tmp_path / "retained parent"

    first = xtb_workflow.run_gfnff(
        _molecule(),
        task=XTBTask.SINGLEPOINT,
        charge=0,
        executable=fake_xtb_factory(directory_name="fake-xtb-first"),
        work_directory=workspace_parent,
        keep_work_directory=True,
    )
    second = xtb_workflow.run_gfnff(
        _molecule(),
        task=XTBTask.SINGLEPOINT,
        charge=0,
        executable=fake_xtb_factory(directory_name="fake-xtb-second"),
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


def test_failed_run_removes_workspace_unless_retention_is_requested(
    fake_xtb_factory: FakeXTBFactory,
    tmp_path: Path,
) -> None:
    workspace_parent = tmp_path / "failed-run-parent"

    with pytest.raises(XTBExecutionError) as error:
        xtb_workflow.run_gfn_xtb(
            _molecule(),
            task=XTBTask.SINGLEPOINT,
            charge=0,
            unpaired_electrons=0,
            executable=fake_xtb_factory("nonzero"),
            work_directory=workspace_parent,
        )

    assert error.value.report.workspace_retained is False
    assert not error.value.report.work_directory.exists()
    assert tuple(workspace_parent.iterdir()) == ()
