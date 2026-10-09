"""Contracts for the five independent force-field benchmark workflows."""

from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pytest

from hotpot.cheminfo import forcefields as ff

from . import (
    auto_optimize_benchmark,
    obwrappers_benchmark,
    openbabel_benchmark,
    optimize_complex_benchmark,
    rdkit_benchmark,
)
from .cohort import LigandCase
from .workflow_runner import (
    _manifest_payload,
    _run_hotpot_target,
    _run_obwrappers_target,
    _run_target,
    _target_summary,
)
from . import workflow_comparison
from .io import sha256_file


@pytest.mark.parametrize(
    ("module", "workflow"),
    (
        (rdkit_benchmark, "rdkit"),
        (openbabel_benchmark, "openbabel"),
        (obwrappers_benchmark, "obwrappers"),
        (optimize_complex_benchmark, "hotpot_optimize_complex"),
        (auto_optimize_benchmark, "hotpot_auto"),
    ),
)
def test_each_launcher_fixes_exactly_one_workflow(
    monkeypatch: pytest.MonkeyPatch,
    module: object,
    workflow: str,
) -> None:
    observed = []

    def fake_main_for_backend(backend, argv):
        observed.append((backend, argv))
        return {"workflow": backend}

    monkeypatch.setattr(module, "main_for_backend", fake_main_for_backend)

    assert module.main(("--help",)) == {"workflow": workflow}
    assert observed == [(workflow, ("--help",))]


class _DummyMolecule:
    def __init__(self) -> None:
        self.writes = []

    def write(self, path, **options) -> None:
        self.writes.append((path, options))


@pytest.mark.parametrize(
    ("backend", "target", "expected_calls"),
    (
        ("hotpot_optimize_complex", "ligand", ("build3d", "optimize")),
        (
            "hotpot_optimize_complex",
            "complex",
            ("build_complex3d", "optimize_complex"),
        ),
        ("hotpot_auto", "ligand", ("build3d", "auto_optimize")),
        ("hotpot_auto", "complex", ("build_complex3d", "auto_optimize")),
    ),
)
def test_hotpot_workflow_keeps_ligand_and_complex_call_paths_distinct(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
    backend: str,
    target: str,
    expected_calls: tuple[str, str],
) -> None:
    calls = []

    def fake_call(name):
        def call(*args, **kwargs):
            calls.append((name, args, kwargs))
            if name.startswith("optimize") or name == "auto_optimize":
                return SimpleNamespace(
                    converged=True,
                    termination_reason="converged",
                    trajectory=None,
                    routing_report=None,
                    quality_report=SimpleNamespace(
                        level="standard",
                        passed=True,
                        checks=(),
                        failures=(),
                        warnings=(),
                        metrics={},
                    ),
                )
            return SimpleNamespace(trajectory=None)

        return call

    for name in (
        "build3d",
        "optimize",
        "build_complex3d",
        "optimize_complex",
        "auto_optimize",
    ):
        monkeypatch.setattr(ff, name, fake_call(name))

    result = _run_hotpot_target(
        backend,
        target,
        _DummyMolecule(),
        tmp_path,
        seed=43,
    )

    assert tuple(name for name, _args, _kwargs in calls) == expected_calls
    optimization_options = calls[-1][2]
    assert optimization_options["convergence_level"] is ff.ConvergenceLevel.FAST
    assert optimization_options["quality_level"] == (
        "standard" if backend == "hotpot_auto" else "off"
    )
    assert result["compute_seconds"] == (
        result["build_seconds"] + result["optimization_seconds"]
    )


def test_obwrappers_workflow_requests_fast_convergence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from hotpot.cheminfo import obWrappers

    observed = []
    monkeypatch.setattr(
        obWrappers,
        "build",
        lambda mol: SimpleNamespace(succeeded=True, rules=()),
    )

    def fake_optimize(mol, forcefield, **options):
        observed.append(options)
        return SimpleNamespace(
            converged=True,
            termination_reason="converged",
            rules=(),
        )

    monkeypatch.setattr(obWrappers, "optimize", fake_optimize)
    monkeypatch.setattr(
        "tests.benchmarks.coordination_complexes.workflow_runner._record_native_frame",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(
        (
            "tests.benchmarks.coordination_complexes.workflow_runner."
            "_record_obwrappers_trajectory"
        ),
        lambda *args, **kwargs: None,
    )

    _run_obwrappers_target(object(), object())

    assert observed[0]["convergence_level"] is ff.ConvergenceLevel.FAST


def test_hotpot_auto_reuses_optimizer_quality_gate(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    molecule = _DummyMolecule()
    molecule.coordinates = np.zeros((1, 3), dtype=float)
    validation = {"level": "standard", "passed": True, "failures": []}
    monkeypatch.setattr(
        "tests.benchmarks.coordination_complexes.workflow_runner._prepare_ligand",
        lambda case: molecule,
    )
    monkeypatch.setattr(ff, "capture_topology", lambda *args, **kwargs: object())
    monkeypatch.setattr(
        ff,
        "evaluate_structure_acceptance",
        lambda *args, **kwargs: pytest.fail("quality gate was evaluated twice"),
    )
    monkeypatch.setattr(
        "tests.benchmarks.coordination_complexes.workflow_runner._run_hotpot_target",
        lambda *args, **kwargs: {
            "compute_seconds": 1.0,
            "validation": validation,
            "quality_passed": True,
        },
    )
    monkeypatch.setattr(
        "tests.benchmarks.coordination_complexes.workflow_runner._write_structure",
        lambda *args, **kwargs: {"mol2": "optimized.mol2"},
    )

    record = _run_target(
        "hotpot_auto",
        "ligand",
        LigandCase(index=1, smiles="N", seed=43),
        None,
        tmp_path,
    )

    assert record["status"] == "passed"
    assert record["validation"] is validation


def test_hotpot_manifests_record_fast_convergence_level() -> None:
    cohort = SimpleNamespace(
        input_path="inputs.smi",
        input_sha256="a" * 64,
        ligand_cases=(object(), object()),
        cohort_path="cohort.json",
        cohort_sha256="b" * 64,
        complex_cases=(object(),),
        complex_indices=(1,),
    )

    for backend in ("obwrappers", "hotpot_optimize_complex", "hotpot_auto"):
        manifest = _manifest_payload(backend, cohort)
        assert manifest["settings"]["convergence_level"] == "FAST"


def test_target_summary_excludes_ineligible_complexes() -> None:
    records = (
        {
            "targets": {
                "complex": {
                    "status": "passed",
                    "quality_passed": True,
                    "compute_seconds": 1.0,
                }
            }
        },
        {
            "targets": {
                "complex": {
                    "status": "failed_quality",
                    "quality_passed": False,
                    "compute_seconds": 3.0,
                }
            }
        },
        {
            "targets": {
                "complex": {
                    "status": "failed_execution",
                    "quality_passed": False,
                    "compute_seconds": 0.1,
                }
            }
        },
        {"targets": {"complex": {"status": "not_eligible"}}},
    )

    summary = _target_summary(records, "complex")

    assert summary["denominator"] == 3
    assert summary["pass_count"] == 1
    assert summary["pass_rate"] == pytest.approx(1.0 / 3.0)
    assert summary["median_compute_seconds"] == 2.0
    assert summary["aggregate_compute_seconds"] == 4.0
    assert summary["timed_count"] == 2


def test_target_summary_reports_auto_route_selection() -> None:
    records = (
        {
            "targets": {
                "complex": {
                    "status": "passed",
                    "quality_passed": True,
                    "compute_seconds": 1.0,
                    "routing_report": {"selected_route": "ordinary_fast"},
                }
            }
        },
        {
            "targets": {
                "complex": {
                    "status": "passed",
                    "quality_passed": True,
                    "compute_seconds": 2.0,
                    "routing_report": {"selected_route": "complex"},
                }
            }
        },
    )

    summary = _target_summary(records, "complex")

    assert summary["selected_route_counts"] == {
        "ordinary_fast": 1,
        "complex": 1,
    }


def test_case_preparation_failure_is_recorded_instead_of_aborting(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    def fail_preparation(case):
        raise RuntimeError(f"cannot prepare case {case.index}")

    monkeypatch.setattr(
        "tests.benchmarks.coordination_complexes.workflow_runner._prepare_ligand",
        fail_preparation,
    )

    record = _run_target(
        "rdkit",
        "ligand",
        LigandCase(index=1, smiles="N", seed=43),
        None,
        tmp_path,
    )

    assert record["status"] == "failed_execution"
    assert record["error_type"] == "RuntimeError"


def test_hotpot_target_writes_built_structure_outside_report_payload(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    molecule = _DummyMolecule()
    monkeypatch.setattr(ff, "build3d", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        ff,
        "optimize",
        lambda *args, **kwargs: SimpleNamespace(
            converged=False,
            termination_reason="limit",
            trajectory=None,
            routing_report=None,
        ),
    )

    _run_hotpot_target(
        "hotpot_optimize_complex",
        "ligand",
        molecule,
        tmp_path,
        seed=43,
    )

    assert len(molecule.writes) == 1
    assert molecule.writes[0][0] == tmp_path / "built.mol2"


def _write_comparison_fixture(root, workflow: str) -> None:
    manifest = {
        "schema_version": 1,
        "workflow": workflow,
        "input": {
            "path": "inputs.smi",
            "sha256": "a" * 64,
            "sample_count": 2,
        },
        "cohort": {
            "path": "cohort.json",
            "sha256": "b" * 64,
            "sample_count": 1,
            "indices": [1],
        },
    }
    if workflow in {"obwrappers", "hotpot_optimize_complex", "hotpot_auto"}:
        manifest["settings"] = {"convergence_level": "FAST"}
    root.mkdir(parents=True)
    (root / "manifest.json").write_text(
        json.dumps(manifest),
        encoding="utf-8",
    )
    summary = {
        "schema_version": 1,
        "workflow": workflow,
        "input_count": 2,
        "complex_eligible_count": 1,
        "manifest_sha256": sha256_file(root / "manifest.json"),
        "targets": {
            "ligand": {
                "denominator": 2,
                "report_count": 2,
                "pass_count": 1,
                "pass_rate": 0.5,
                "median_compute_seconds": 2.0,
                "aggregate_compute_seconds": 4.0,
                "timed_count": 2,
                "status_counts": {"passed": 1, "failed_quality": 1},
            },
            "complex": {
                "denominator": 1,
                "report_count": 1,
                "pass_count": 1,
                "pass_rate": 1.0,
                "median_compute_seconds": 2.0,
                "aggregate_compute_seconds": 2.0,
                "timed_count": 1,
                "status_counts": {"passed": 1},
            },
        },
    }
    (root / "summary.json").write_text(
        json.dumps(summary),
        encoding="utf-8",
    )
    for index, ligand_status in ((1, "passed"), (2, "failed_quality")):
        case_dir = root / "cases" / f"{index:04d}"
        case_dir.mkdir(parents=True)
        ligand_passed = ligand_status == "passed"
        complex_target = (
            {
                "status": "passed",
                "quality_passed": True,
                "compute_seconds": 2.0,
                "validation": {"failures": []},
            }
            if index == 1
            else {"status": "not_eligible"}
        )
        report = {
            "schema_version": 1,
            "workflow": workflow,
            "index": index,
            "smiles": "N" if index == 1 else "O",
            "targets": {
                "ligand": {
                    "status": ligand_status,
                    "quality_passed": ligand_passed,
                    "compute_seconds": 1.0 if index == 1 else 3.0,
                    "validation": {
                        "failures": [] if ligand_passed else [{"name": "gate"}]
                    },
                },
                "complex": complex_target,
            },
        }
        (case_dir / "report.json").write_text(
            json.dumps(report),
            encoding="utf-8",
        )


def test_workflow_comparison_aggregates_both_targets(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    monkeypatch.setattr(workflow_comparison, "INPUT_SAMPLE_COUNT", 2)
    monkeypatch.setattr(workflow_comparison, "COMPLEX_SAMPLE_COUNT", 1)
    roots = {}
    for workflow in workflow_comparison.WORKFLOWS:
        root = tmp_path / workflow
        _write_comparison_fixture(root, workflow)
        roots[workflow] = root

    output = tmp_path / "assets"
    payload = workflow_comparison.aggregate_workflow_comparison(
        roots["rdkit"],
        roots["openbabel"],
        roots["obwrappers"],
        roots["hotpot_optimize_complex"],
        roots["hotpot_auto"],
        output,
    )

    assert len(payload["results"]) == 10
    assert {row["target"] for row in payload["results"]} == {
        "ligand",
        "complex",
    }
    assert (output / "coordination_complex_backend_comparison.csv").is_file()
    assert (output / "coordination_complex_backend_comparison_cases.csv").is_file()
    assert (output / "coordination_complex_backend_comparison.png").is_file()


def test_workflow_comparison_rejects_non_fast_hotpot_manifest(tmp_path) -> None:
    root = tmp_path / "obwrappers"
    _write_comparison_fixture(root, "obwrappers")
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["settings"]["convergence_level"] = "STRICT"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="must use FAST convergence"):
        workflow_comparison._validate_run(root, "obwrappers")
