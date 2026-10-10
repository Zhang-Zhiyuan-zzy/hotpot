"""Contracts for the opt-in xTB coordination benchmark."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from hotpot import read_mol
from hotpot.cheminfo.calculator.electronic_state import (
    ChargeInferenceSource,
    SpinInferenceSource,
)
from hotpot.plugins.xtb import XTBMethod, XTBTask

from .cohort import BenchmarkCohort, LigandCase
from .io import write_json
from .xtb_benchmark import (
    BenchmarkTarget,
    XTBBenchmarkConfig,
    XTBRoute,
    _route_summary,
    _run_route,
    _thread_environment,
    _validate_source_identity,
    _write_or_check_manifest,
    parse_case_selection,
    run_case,
)


def _cohort(tmp_path: Path) -> BenchmarkCohort:
    input_path = tmp_path / "inputs.smi"
    cohort_path = tmp_path / "canonical_cases.json"
    input_path.write_text("O\nN\n", encoding="utf-8")
    cohort_path.write_text("{}\n", encoding="utf-8")
    return BenchmarkCohort(
        input_path=input_path,
        input_sha256="a" * 64,
        cohort_path=cohort_path,
        cohort_sha256="b" * 64,
        ligand_cases=(
            LigandCase(index=1, smiles="O", seed=11),
            LigandCase(index=2, smiles="N", seed=12),
        ),
        complex_cases={},
    )


def _water():
    mol = read_mol("[H]O[H]", fmt="smi")
    mol.coordinates = np.asarray(
        ((-0.76, 0.58, 0.0), (0.0, 0.0, 0.0), (0.76, 0.58, 0.0)),
        dtype=float,
    )
    return mol


def _fake_report(method: XTBMethod):
    return SimpleNamespace(
        backend_info=SimpleNamespace(
            executable=Path("/opt/xtb/bin/xtb"),
            version="6.7.1",
            revision="test",
            executable_sha256="c" * 64,
        ),
        requested_method=method,
        effective_method=method,
        task=XTBTask.OPTIMIZE,
        charge=0,
        unpaired_electrons=None if method is XTBMethod.GFNFF else 0,
        charge_source=ChargeInferenceSource.VALENCE,
        spin_source=(
            None
            if method is XTBMethod.GFNFF
            else SpinInferenceSource.LOWEST_SPIN_PARITY
        ),
        state_assumptions=(),
        fragment_charges=(0,),
        argv=("xtb", "input.xyz"),
        return_code=0,
        elapsed_seconds=0.25,
        process_succeeded=True,
        converged=True,
        energy_hartree=-1.0,
        gradient_norm=0.01,
        atom_order_verified=True,
        coordinates_committed=True,
        workspace_retained=True,
        artifacts={},
        stdout="native stdout",
        stderr="native stderr",
    )


@pytest.mark.parametrize(
    ("text", "expected"),
    (
        ("1", (1,)),
        ("1-3,5,3", (1, 2, 3, 5)),
        (" 2-4, 8 ", (2, 3, 4, 8)),
    ),
)
def test_case_selection_accepts_indices_and_inclusive_ranges(
    text: str,
    expected: tuple[int, ...],
) -> None:
    assert parse_case_selection(text) == expected


@pytest.mark.parametrize("text", ("", "0", "4-2"))
def test_case_selection_rejects_empty_nonpositive_and_descending_ranges(
    text: str,
) -> None:
    with pytest.raises(argparse.ArgumentTypeError):
        parse_case_selection(text)


def test_source_identity_requires_the_same_input_and_cohort_hashes(
    tmp_path: Path,
) -> None:
    cohort = _cohort(tmp_path)
    matching = {
        "input": {"sha256": cohort.input_sha256},
        "cohort": {"sha256": cohort.cohort_sha256},
    }

    _validate_source_identity(matching, cohort)

    with pytest.raises(ValueError, match="ligand corpus"):
        _validate_source_identity(
            {**matching, "input": {"sha256": "0" * 64}},
            cohort,
        )
    with pytest.raises(ValueError, match="canonical cohort"):
        _validate_source_identity(
            {**matching, "cohort": {"sha256": "0" * 64}},
            cohort,
        )


def test_resume_accepts_only_the_same_manifest_configuration(
    tmp_path: Path,
) -> None:
    output_root = tmp_path / "output"
    output_root.mkdir()
    first = {"schema_version": 1, "created_at": "first", "settings": {"x": 1}}
    second = {"schema_version": 1, "created_at": "second", "settings": {"x": 1}}
    changed = {"schema_version": 1, "created_at": "third", "settings": {"x": 2}}

    _write_or_check_manifest(output_root, first, resume=False)
    _write_or_check_manifest(output_root, second, resume=True)

    with pytest.raises(ValueError, match="configuration differs"):
        _write_or_check_manifest(output_root, changed, resume=True)
    with pytest.raises(FileExistsError):
        _write_or_check_manifest(output_root, first, resume=False)


def test_fake_routes_persist_stage_reports_logs_structures_and_quality(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    import tests.benchmarks.coordination_complexes.xtb_benchmark as benchmark

    calls = []

    def fake_execute(
        mol,
        method,
        charge_state,
        electronic_state,
        config,
        work_directory,
    ):
        calls.append((method, np.asarray(mol.coordinates, dtype=float).copy()))
        mol.coordinates = np.asarray(mol.coordinates, dtype=float) + 0.01
        return _fake_report(method)

    monkeypatch.setattr(benchmark, "_execute_xtb_stage", fake_execute)
    monkeypatch.setattr(
        benchmark,
        "_evaluate_quality",
        lambda mol, quality_level, topology_reference: {
            "level": quality_level,
            "passed": True,
            "checks": (),
            "failures": (),
            "warnings": (),
            "metrics": {},
        },
    )
    source_mol = _water()
    charge_state = benchmark.infer_charge(source_mol)
    electronic_state = benchmark.resolve_electronic_state(source_mol)
    config = XTBBenchmarkConfig(
        routes=(XTBRoute.DIRECT_GFN2, XTBRoute.GFNFF_GFN2),
        targets=(BenchmarkTarget.LIGAND,),
        workers=1,
        threads=4,
    )

    direct = _run_route(
        source_mol,
        XTBRoute.DIRECT_GFN2,
        charge_state,
        electronic_state,
        config,
        tmp_path / "direct",
    )
    chain = _run_route(
        source_mol,
        XTBRoute.GFNFF_GFN2,
        charge_state,
        electronic_state,
        config,
        tmp_path / "chain",
    )

    assert direct["status"] == chain["status"] == "passed"
    assert [call[0] for call in calls] == [
        XTBMethod.GFN2_XTB,
        XTBMethod.GFNFF,
        XTBMethod.GFN2_XTB,
    ]
    np.testing.assert_allclose(calls[0][1], calls[1][1])
    np.testing.assert_allclose(calls[2][1], calls[1][1] + 0.01)
    assert (tmp_path / "direct/00-gfn2/report.json").is_file()
    assert (tmp_path / "direct/00-gfn2/native.log").read_text(
        encoding="utf-8"
    ).startswith("=== stdout ===\nnative stdout")
    assert (tmp_path / "direct/00-gfn2/structure.sdf").is_file()
    assert (tmp_path / "direct/final.sdf").is_file()
    assert (tmp_path / "chain/00-gfnff/report.json").is_file()
    assert (tmp_path / "chain/01-gfn2/report.json").is_file()
    assert json.loads(
        (tmp_path / "chain/report.json").read_text(encoding="utf-8")
    )["quality"]["passed"] is True
    assert _thread_environment(4)["OMP_NUM_THREADS"] == "4"


def test_route_summary_keeps_target_denominators_and_missing_sources_visible() -> None:
    records = (
        {
            "targets": {
                "ligand": {
                    "status": "completed",
                    "routes": {
                        "direct-gfn2": {
                            "status": "passed",
                            "total_seconds": 1.0,
                            "applicability_rejected": False,
                            "stages": (
                                {"method": "gfn2", "elapsed_seconds": 0.8},
                            ),
                        }
                    },
                },
                "complex": {
                    "status": "completed",
                    "routes": {
                        "direct-gfn2": {
                            "status": "failed_quality",
                            "total_seconds": 2.0,
                            "applicability_rejected": False,
                            "stages": (
                                {"method": "gfn2", "elapsed_seconds": 1.7},
                            ),
                        }
                    },
                },
            }
        },
        {
            "targets": {
                "ligand": {"status": "source_unavailable", "routes": {}},
                "complex": {"status": "not_eligible", "routes": {}},
            }
        },
    )

    ligand = _route_summary(
        records,
        BenchmarkTarget.LIGAND,
        XTBRoute.DIRECT_GFN2,
    )
    complex_summary = _route_summary(
        records,
        BenchmarkTarget.COMPLEX,
        XTBRoute.DIRECT_GFN2,
    )

    assert ligand["denominator"] == 2
    assert ligand["source_available_count"] == 1
    assert ligand["quality_pass_count"] == 1
    assert ligand["quality_pass_rate"] == pytest.approx(0.5)
    assert complex_summary["denominator"] == 1
    assert complex_summary["quality_pass_count"] == 0
    assert complex_summary["status_counts"] == {"failed_quality": 1}


def test_completed_case_report_is_the_resume_boundary(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    output_root = tmp_path / "output"
    report_path = output_root / "cases/0001/report.json"
    report_path.parent.mkdir(parents=True)
    expected = {
        "schema_version": 1,
        "index": 1,
        "smiles": "O",
        "workflow": "xtb_coordination_refinement",
        "targets": {},
        "total_seconds": 1.0,
    }
    write_json(report_path, expected)
    monkeypatch.setattr(
        "tests.benchmarks.coordination_complexes.xtb_benchmark._target_record",
        lambda *args, **kwargs: pytest.fail("resume recomputed a completed case"),
    )

    observed = run_case(
        {"index": 1, "smiles": "O", "seed": 11},
        None,
        str(tmp_path / "source"),
        str(output_root),
        XTBBenchmarkConfig(workers=1),
        True,
    )

    assert observed == expected
