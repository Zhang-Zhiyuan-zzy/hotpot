"""Strict contracts for thin CBond, force-field and xTB stage adapters."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest


def _water():
    import hotpot

    mol = hotpot.read_mol("[H]O[H]", "smi")
    mol.coordinates = np.asarray(
        (
            (-0.75, 0.0, 0.0),
            (0.0, 0.5, 0.0),
            (0.75, 0.0, 0.0),
        )
    )
    return mol


def _context(stage_spec, tmp_path: Path):
    from hotpot.pipeline.contracts import StageContext

    run_directory = tmp_path / "run"
    stage_directory = run_directory / "stages" / ".00-stage.tmp-test"
    stage_directory.mkdir(parents=True)
    return StageContext(run_directory, stage_directory, 0, stage_spec)


@dataclass(frozen=True)
class _QualityReport:
    passed: bool


@dataclass(frozen=True)
class _ForceFieldReport:
    quality_report: _QualityReport
    trajectory: None
    requested_forcefield: str
    effective_forcefield: str


def test_cbond_stage_calls_detailed_public_result_api_once(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from hotpot.cheminfo.AImodels.cbond import apply
    from hotpot.cheminfo.AImodels.cbond import stage as cbond_stage
    from hotpot.pipeline.contracts import MolecularPayload, StageSpec, StageStatus

    output_mol = _water()
    calls: list[tuple[str, str, bool]] = []

    def auto_build_cbond(mol, metal, **options):
        calls.append((mol.smiles, str(metal), options["return_details"]))
        return SimpleNamespace(
            molecule=output_mol,
            steps=(),
            donor_indices=(),
            path_probability=1.0,
        )

    monkeypatch.setattr(apply, "auto_build_cbond", auto_build_cbond)
    monkeypatch.setattr(apply, "get_cbond_runtime", lambda *args: object())
    spec = StageSpec("cbond", ("Eu", "NCCO", "--device", "cpu"))
    prepared = cbond_stage.get_stage().prepare(spec)

    result = prepared.execute(MolecularPayload(()), _context(spec, tmp_path))

    assert len(calls) == 1
    assert calls[0][1:] == ("Eu", True)
    assert result.status is StageStatus.SUCCEEDED
    assert len(result.payload.records) == 1
    assert result.payload.records[0].molecule is output_mol
    assert "cbond" in result.report


def test_cbond_stage_emits_all_structures_as_molecular_records(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from hotpot.cheminfo.AImodels.cbond import apply
    from hotpot.cheminfo.AImodels.cbond import stage as cbond_stage
    from hotpot.pipeline.contracts import MolecularPayload, StageSpec

    structures = (_water(), _water())
    calls: list[int] = []

    def build_all_possible_cbond(mol, metal, **options):
        calls.append(options["max_states"])
        return tuple(
            SimpleNamespace(
                molecule=structure,
                probability=0.75 - index * 0.5,
                steps=(),
                donor_indices=(index,),
                path_count=1,
            )
            for index, structure in enumerate(structures)
        )

    monkeypatch.setattr(
        apply,
        "build_all_possible_cbond",
        build_all_possible_cbond,
    )
    monkeypatch.setattr(apply, "get_cbond_runtime", lambda *args: object())
    spec = StageSpec(
        "cbond",
        ("Eu", "NCCO", "--all-structures", "--max-states", "7"),
    )

    result = cbond_stage.get_stage().prepare(spec).execute(
        MolecularPayload(()),
        _context(spec, tmp_path),
    )

    assert calls == [7]
    assert tuple(record.molecule for record in result.payload.records) == structures
    assert tuple(
        structure["rank"] for structure in result.report["cbond"]["structures"]
    ) == (1, 2)


def test_forcefield_stage_uses_one_public_route_operation_without_duplicate_gate(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    import hotpot.cheminfo.forcefields as ff
    from hotpot.cheminfo.forcefields import acceptance
    from hotpot.cheminfo.forcefields import stage as ff_stage
    from hotpot.pipeline.contracts import (
        MolecularPayload,
        MolecularRecord,
        StageSpec,
        StageStatus,
    )

    calls = 0

    def optimize(mol, forcefield=None, **options):
        nonlocal calls
        calls += 1
        return _ForceFieldReport(
            quality_report=_QualityReport(passed=True),
            trajectory=None,
            requested_forcefield=forcefield or "UFF",
            effective_forcefield="UFF",
        )

    def duplicate_gate(*args, **kwargs):
        raise AssertionError("the stage repeated a geometry gate owned by the FF route")

    monkeypatch.setattr(ff, "optimize", optimize)
    monkeypatch.setattr(ff_stage, "optimize", optimize, raising=False)
    monkeypatch.setattr(
        acceptance,
        "evaluate_structure_acceptance",
        duplicate_gate,
    )
    monkeypatch.setattr(
        ff_stage,
        "evaluate_structure_acceptance",
        duplicate_gate,
        raising=False,
    )
    spec = StageSpec(
        "ff",
        (
            "--route",
            "organic",
            "--optimize-only",
            "--forcefield",
            "uff",
            "--quality",
            "standard",
        ),
    )
    mol = _water()
    payload = MolecularPayload((MolecularRecord(mol),))
    prepared = ff_stage.get_stage().prepare(spec)

    result = prepared.execute(payload, _context(spec, tmp_path))

    assert calls == 1
    assert result.status is StageStatus.SUCCEEDED
    assert result.payload.records[0].molecule is mol
    assert "forcefield" in result.report


@dataclass(frozen=True)
class _XTBReport:
    process_succeeded: bool = True
    converged: bool = True
    coordinates_committed: bool = True
    energy_hartree: float = -76.0
    stderr: str = ""


def test_xtb_stage_forwards_existing_state_and_calls_public_workflow_once(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from hotpot.cheminfo.calculator.electronic_state import (
        ChargeInferenceSource,
        ElectronicState,
        SpinInferenceSource,
    )
    from hotpot.cheminfo.forcefields import acceptance
    from hotpot.pipeline.contracts import (
        MolecularPayload,
        MolecularRecord,
        StageSpec,
        StageStatus,
    )
    from hotpot.plugins.xtb import stage as xtb_stage
    from hotpot.plugins.xtb import workflow

    state = ElectronicState(
        charge=0,
        unpaired_electrons=0,
        multiplicity=1,
        fragment_charges=(0,),
        charge_source=ChargeInferenceSource.EXPLICIT,
        spin_source=SpinInferenceSource.EXPLICIT,
        assumptions=("explicit test state",),
    )
    calls: list[tuple[object, object]] = []

    def run_gfn_xtb(mol, **options):
        calls.append((mol, options["state"]))
        return _XTBReport()

    def duplicate_gate(*args, **kwargs):
        raise AssertionError("the stage repeated a geometry gate owned by xTB")

    monkeypatch.setattr(workflow, "run_gfn_xtb", run_gfn_xtb)
    monkeypatch.setattr(xtb_stage, "run_gfn_xtb", run_gfn_xtb, raising=False)
    monkeypatch.setattr(
        acceptance,
        "evaluate_structure_acceptance",
        duplicate_gate,
    )
    monkeypatch.setattr(
        xtb_stage,
        "evaluate_structure_acceptance",
        duplicate_gate,
        raising=False,
    )
    spec = StageSpec(
        "xtb",
        ("--method", "gfn2", "--task", "optimize", "--post-check", "off"),
    )
    mol = _water()
    payload = MolecularPayload((MolecularRecord(mol, state),))
    prepared = xtb_stage.get_stage().prepare(spec)

    result = prepared.execute(payload, _context(spec, tmp_path))

    assert calls == [(mol, state)]
    assert result.status is StageStatus.SUCCEEDED
    assert result.payload.records[0].molecule is mol
    assert result.payload.records[0].electronic_state is state
    assert result.report["xtb"]["energy_hartree"] == pytest.approx(-76.0)


def test_gfnff_stage_calls_only_gfnff_node(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from hotpot.pipeline.contracts import (
        MolecularPayload,
        MolecularRecord,
        StageSpec,
        StageStatus,
    )
    from hotpot.plugins.xtb import stage as xtb_stage
    from hotpot.plugins.xtb import workflow

    calls: list[str] = []

    def run_gfnff(mol, **options):
        calls.append("gfnff")
        return _XTBReport()

    def must_not_call_gfn_xtb(mol, **options):
        calls.append("gfn-xtb")
        raise AssertionError("the independent GFN-FF stage called GFN-xTB")

    monkeypatch.setattr(workflow, "run_gfnff", run_gfnff)
    monkeypatch.setattr(workflow, "run_gfn_xtb", must_not_call_gfn_xtb)
    monkeypatch.setattr(xtb_stage, "run_gfnff", run_gfnff, raising=False)
    monkeypatch.setattr(
        xtb_stage,
        "run_gfn_xtb",
        must_not_call_gfn_xtb,
        raising=False,
    )
    spec = StageSpec("xtb", ("--method", "gfnff", "--task", "optimize"))
    mol = _water()
    prepared = xtb_stage.get_stage().prepare(spec)

    result = prepared.execute(
        MolecularPayload((MolecularRecord(mol),)),
        _context(spec, tmp_path),
    )

    assert calls == ["gfnff"]
    assert result.status is StageStatus.SUCCEEDED
    state = result.payload.records[0].electronic_state
    assert state is not None
    assert state.charge == 0
    assert state.unpaired_electrons == 0

