"""Contracts for the five independent force-field benchmark workflows."""

from __future__ import annotations

import json
from types import SimpleNamespace

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
    _run_hotpot_auto_target,
    _run_hotpot_target,
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
    ("target", "expected_calls"),
    (
        ("ligand", ("build3d", "optimize")),
        ("complex", ("build_complex3d", "optimize_complex")),
    ),
)
def test_hotpot_workflow_keeps_ligand_and_complex_call_paths_distinct(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
    target: str,
    expected_calls: tuple[str, str],
) -> None:
    calls = []

    def fake_call(name):
        def call(*args, **kwargs):
            calls.append((name, args, kwargs))
            if name.startswith("optimize"):
                return SimpleNamespace(
                    converged=True,
                    termination_reason="converged",
                    trajectory=None,
                )
            return SimpleNamespace(trajectory=None)

        return call

    for name in ("build3d", "optimize", "build_complex3d", "optimize_complex"):
        monkeypatch.setattr(ff, name, fake_call(name))

    result = _run_hotpot_target(
        target,
        _DummyMolecule(),
        tmp_path,
        seed=43,
    )

    assert tuple(name for name, _args, _kwargs in calls) == expected_calls
    assert result["compute_seconds"] == (
        result["build_seconds"] + result["optimization_seconds"]
    )


def test_hotpot_auto_complex_starts_without_preliminary_complex_build(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    calls = []
    quality = SimpleNamespace(
        level="standard",
        passed=True,
        checks=(),
        failures=(),
        warnings=(),
        metrics={},
    )
    routing = ff.OptimizationRoutingReport(
        selected_route=ff.OptimizationRoute.NATIVE_FAST,
        attempts=(),
    )

    def auto_optimize(molecule, forcefield, **options):
        calls.append((molecule, forcefield, options))
        return SimpleNamespace(
            converged=True,
            termination_reason="converged",
            quality_report=quality,
            routing_report=routing,
            trajectory=None,
        )

    monkeypatch.setattr(ff, "auto_optimize", auto_optimize)
    monkeypatch.setattr(
        ff,
        "build_complex3d",
        lambda *args, **kwargs: pytest.fail("auto benchmark prebuilt the complex"),
    )

    result = _run_hotpot_auto_target(
        "complex",
        _DummyMolecule(),
        tmp_path,
        seed=43,
    )

    assert len(calls) == 1
    assert calls[0][2]["convergence_level"] is ff.ConvergenceLevel.FAST
    assert calls[0][2]["trajectory_start"] is ff.TrajectoryStart.LIGAND_BUILD
    assert result["quality_passed"] is True
    assert result["routing_report"]["selected_route"] == "native_fast"


def test_hotpot_manifests_describe_fast_auto_fallback() -> None:
    cohort = SimpleNamespace(
        input_path="inputs.smi",
        input_sha256="a" * 64,
        ligand_cases=(object(), object()),
        cohort_path="cohort.json",
        cohort_sha256="b" * 64,
        complex_cases=(object(),),
        complex_indices=(1,),
    )

    manifest = _manifest_payload("hotpot_auto", cohort)

    assert manifest["settings"]["convergence_level"] == "FAST"
    assert manifest["settings"]["strategy"] == (
        "obwrappers_fast_then_complex_workflow"
    )


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
                    "routing_report": {"selected_route": "native_fast"},
                }
            }
        },
        {
            "targets": {
                "complex": {
                    "status": "passed",
                    "quality_passed": True,
                    "compute_seconds": 2.0,
                    "routing_report": {"selected_route": "complex_workflow"},
                }
            }
        },
    )

    summary = _target_summary(records, "complex")

    assert summary["selected_route_counts"] == {
        "native_fast": 1,
        "complex_workflow": 1,
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
        ),
    )

    _run_hotpot_target("ligand", molecule, tmp_path, seed=43)

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
    if workflow in workflow_comparison.FAST_WORKFLOWS:
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


def test_workflow_comparison_rejects_nonfast_hotpot_manifest(tmp_path) -> None:
    root = tmp_path / "obwrappers"
    _write_comparison_fixture(root, "obwrappers")
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["settings"]["convergence_level"] = "STRICT"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="must use FAST convergence"):
        workflow_comparison._validate_run(root, "obwrappers")
