"""Pre-activation scientific-state resolver benchmark tests."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import hotpot
import pytest

from hotpot.cheminfo.core import BondKind

from .corpus import DryStateCase, DryStateCorpus, DryStateSourceKind
from .runner import DryCaseError, DryResolutionRecord, resolve_case, run_benchmark


def _case(smiles: str = "[Eu].N") -> DryStateCase:
    return DryStateCase(
        index=1,
        case_id="eu-n",
        source_kind=DryStateSourceKind.SMILES,
        source=smiles,
        format="smi",
        source_sha256=hashlib.sha256(smiles.encode("utf-8")).hexdigest(),
    )


def test_resolve_case_records_only_the_real_profile_blockers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import tests.benchmarks.scientific_state.runner as runner

    mol = hotpot.read_mol("[Eu].N", "smi")
    mol.add_bond(0, 1, bond_kind=BondKind.DATIVE)
    monkeypatch.setattr(runner, "_load_molecule", lambda case: mol)

    record = resolve_case(_case())

    assert record.profile_activation_status == "planned"
    assert record.coordination_bond_count == 1
    assert record.profile_in_domain is False
    assert record.profile_finding_codes == (
        "PROFILE.SCREENING_ARTIFACT_NOT_ACTIVE",
    )
    assert record.static_profile_ready_except_activation is True
    assert record.python_peak_bytes > 0
    assert len(record.snapshot_sha256) == 64


def test_unknown_metal_bond_remains_an_explicit_profile_boundary() -> None:
    record = resolve_case(_case("[Eu]N"))

    assert record.coordination_bond_count == 0
    assert "PROFILE.COORDINATION_BOUNDARY_REQUIRED" in (
        record.profile_finding_codes
    )
    assert record.static_profile_ready_except_activation is False


def test_resolve_case_rejects_changed_structure_asset(tmp_path: Path) -> None:
    source = tmp_path / "structure.xyz"
    source.write_text("1\nrecord\nH 0 0 0\n", encoding="utf-8")
    case = DryStateCase(
        index=2,
        case_id="changed",
        source_kind=DryStateSourceKind.FILE,
        source=str(source),
        format="xyz",
        source_sha256="0" * 64,
    )

    with pytest.raises(DryCaseError, match="source: ValueError") as error:
        resolve_case(case)
    assert error.value.stage == "source"


def _record(case: DryStateCase, snapshot: str) -> DryResolutionRecord:
    return DryResolutionRecord(
        index=case.index,
        case_id=case.case_id,
        source_kind=case.source_kind.value,
        format=case.format,
        source_sha256=case.source_sha256,
        facts_identity=f"facts-{case.case_id}",
        atom_count=2,
        coordination_bond_count=1,
        profile_id="coordination_screening_v1",
        profile_version="1.0.0",
        profile_activation_status="planned",
        profile_content_digest="profile",
        parse_seconds=0.1,
        facts_seconds=0.2,
        resolve_seconds=0.5,
        python_peak_bytes=1024 * case.index,
        expanded_path_count=case.index,
        rejected_path_count=0,
        frontier_count=0,
        truncated=case.index == 2,
        search_status="no_certified_decision",
        scientific_resolution_status="ambiguous",
        scientific_resolution_exhaustive=True,
        decision_action=(
            "no_automatic_state" if case.index == 1 else "inadmissible"
        ),
        decision_source="not_selected",
        certification_status="non_equivalent_tie",
        profile_in_domain=False,
        profile_finding_codes=("PROFILE.SCREENING_ARTIFACT_NOT_ACTIVE",),
        static_profile_ready_except_activation=True,
        snapshot_sha256=snapshot,
    )


def test_run_benchmark_streams_records_summary_and_replay(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    import tests.benchmarks.scientific_state.runner as runner

    cases = (_case("O"), _case("N"))
    cases = (
        cases[0],
        DryStateCase(
            index=2,
            case_id="nitrogen",
            source_kind=DryStateSourceKind.SMILES,
            source=cases[1].source,
            format="smi",
            source_sha256=cases[1].source_sha256,
        ),
    )
    corpus = DryStateCorpus(tmp_path / "input.smi", "corpus", cases)
    snapshots = iter(("a", "a", "b", "changed"))
    monkeypatch.setattr(
        runner,
        "resolve_case",
        lambda case: _record(case, next(snapshots)),
    )
    monkeypatch.setattr(
        runner,
        "_coordination_screening_manifest",
        lambda: SimpleNamespace(
            profile_id="coordination_screening_v1",
            profile_version="1.0.0",
            activation_status="planned",
            profile_content_digest="profile",
        ),
    )

    summary = run_benchmark(corpus, tmp_path / "output")

    assert summary["resolved_count"] == 2
    assert summary["mean_expanded_paths"] == pytest.approx(1.5)
    assert summary["max_expanded_paths"] == 2
    assert summary["truncated_fraction"] == pytest.approx(0.5)
    assert summary["automatic_selection_fraction"] == pytest.approx(0.0)
    assert summary["replay_checked_count"] == 2
    assert summary["replay_mismatch_count"] == 1
    records = tuple(
        json.loads(line)
        for line in (tmp_path / "output/records.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    )
    assert tuple(record["case_id"] for record in records) == (
        "eu-n",
        "nitrogen",
    )
    assert records[0]["replay"]["match"] is True
    assert records[1]["replay"]["match"] is False
    assert json.loads(
        (tmp_path / "output/summary.json").read_text(encoding="utf-8")
    )["activation_coverage_is_final"] is False
