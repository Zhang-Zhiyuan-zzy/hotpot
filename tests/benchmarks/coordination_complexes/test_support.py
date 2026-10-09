"""Fast tests for benchmark contracts; no force-field benchmark is run."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from .cli import build_parser
from .configuration import (
    BUILTIN_BACKENDS,
    BUILTIN_SUITES,
    SMOKE_SETTINGS,
    STANDARD_SETTINGS,
    RunProfile,
)
from .io import load_smiles
from .optimizer_comparison import (
    WORKFLOW as OPTIMIZER_COMPARISON_WORKFLOW,
    _clone_starting_mol,
    _run_optimizer_arm,
    _starting_point_fingerprint,
    aggregate_optimizer_comparison,
    build_parser as build_optimizer_comparison_parser,
    run_optimizer_comparison_case,
)
from .rendering import render_experiment
from .reporting import aggregate_run
from .pipeline import (
    CASE_RUNNERS,
    _infer_cbond,
    _run_forcefield_backend,
    _trajectory_payload,
    _write_quality_failure_frame,
    run_hotpot_auto_case,
)
from .runner import _scientific_configuration, _write_or_check_manifest


def test_builtin_extractant_suite_has_187_records() -> None:
    suite = BUILTIN_SUITES["extractants-eu-187"]
    records = load_smiles(suite.input_path)

    assert len(records) == suite.expected_count == 187
    assert records[0][0] == 1
    assert records[-1][0] == 187


def test_builtin_am_suite_reuses_all_187_extractants() -> None:
    suite = BUILTIN_SUITES["extractants-am-187"]

    assert suite.metal == "Am"
    assert suite.expected_count == 187
    assert load_smiles(suite.input_path) == load_smiles(
        BUILTIN_SUITES["extractants-eu-187"].input_path
    )


def test_smoke_profile_is_explicitly_smaller_than_standard() -> None:
    assert SMOKE_SETTINGS.epochs < STANDARD_SETTINGS.epochs
    assert SMOKE_SETTINGS.steps_per_epoch < STANDARD_SETTINGS.steps_per_epoch
    assert SMOKE_SETTINGS.max_attempts < STANDARD_SETTINGS.max_attempts


def test_cbond_threshold_policy_is_explicit_in_settings_and_manifest() -> None:
    assert STANDARD_SETTINGS.first_cbond_threshold == -0.5
    assert STANDARD_SETTINGS.subsequent_cbond_threshold == -0.125

    suite = BUILTIN_SUITES["extractants-eu-187"]
    manifest = _scientific_configuration(
        suite,
        STANDARD_SETTINGS,
        RunProfile.STANDARD,
        "synthetic-sha256",
        (1,),
        "hotpot",
    )

    assert manifest["workflow"] == "cbond-complexes-build"
    assert manifest["settings"]["first_cbond_threshold"] == -0.5
    assert manifest["settings"]["subsequent_cbond_threshold"] == -0.125


def test_am_automatic_manifest_has_configured_backend_and_workflow() -> None:
    manifest = _scientific_configuration(
        BUILTIN_SUITES["extractants-am-187"],
        STANDARD_SETTINGS,
        RunProfile.STANDARD,
        "synthetic-sha256",
        (1,),
        "hotpot-auto",
    )

    assert manifest["backend"] == "hotpot-auto"
    assert manifest["workflow"] == "cbond-auto-optimize"
    assert manifest["backend_configuration"] == (
        BUILTIN_BACKENDS["hotpot-auto"].to_manifest()
    )
    assert manifest["suite"]["metal"] == "Am"
    assert CASE_RUNNERS["hotpot-auto"].__name__ == "run_hotpot_auto_case"


def test_hotpot_auto_backend_calls_auto_directly_without_external_gate(
    tmp_path: Path,
    monkeypatch,
) -> None:
    from hotpot.cheminfo import forcefields as ff

    calls = []
    quality_report = _SyntheticQualityReport()
    trajectory = object()
    routing_report = SimpleNamespace(attempts=(), selected_route="native_fast")
    optimization_report = SimpleNamespace(
        requested_forcefield="UFF",
        effective_forcefield="UFF",
        quality_report=quality_report,
        trajectory=trajectory,
        routing_report=routing_report,
    )

    def fake_auto_optimize(mol, forcefield, **options):
        calls.append((mol, forcefield, options))
        return optimization_report

    def forbidden(*args, **kwargs):
        raise AssertionError("automatic benchmark must not prebuild or re-gate")

    monkeypatch.setattr(ff, "auto_optimize", fake_auto_optimize)
    monkeypatch.setattr(ff, "complexes_build", forbidden)
    monkeypatch.setattr(ff, "evaluate_structure_acceptance", forbidden)
    molecule = object()
    outcome = _run_forcefield_backend(
        molecule,
        BUILTIN_BACKENDS["hotpot-auto"],
        SMOKE_SETTINGS,
        7,
    )

    assert outcome.optimization is optimization_report
    assert outcome.quality_report is quality_report
    assert outcome.routing_report is routing_report
    assert len(calls) == 1
    assert calls[0][0] is molecule
    assert calls[0][1] == "UFF"
    assert calls[0][2]["convergence_level"] is ff.ConvergenceLevel.FAST
    assert calls[0][2]["trajectory_path"] is None
    assert calls[0][2]["save_movie"] is True


def test_quality_failure_exports_last_finite_frame_when_selection_is_nonfinite(
    tmp_path: Path,
) -> None:
    from hotpot import read_mol
    from hotpot.cheminfo import forcefields as ff

    molecule = read_mol("N", fmt="smi")
    molecule.coordinates = np.asarray(((0.0, 0.0, 0.0),), dtype=float)
    trajectory = ff.ForceFieldTrajectory.from_molecule(
        molecule,
        start=ff.TrajectoryStart.FINAL_OPTIMIZATION,
    )
    finite_frame = trajectory.record_molecule(
        molecule,
        stage=ff.TrajectoryStage.FINAL_OPTIMIZATION,
        event=ff.TrajectoryEvent.INITIAL,
    )
    molecule.coordinates = np.asarray(((np.nan, 0.0, 0.0),), dtype=float)
    nonfinite_frame = trajectory.record_molecule(
        molecule,
        stage=ff.TrajectoryStage.FINAL_OPTIMIZATION,
        event=ff.TrajectoryEvent.EPOCH_COMPLETE,
    )
    trajectory.select(nonfinite_frame.index)
    trajectory.set_terminal(nonfinite_frame.index)
    archive = ff.ForceFieldTrajectoryArchive(main=trajectory)

    evidence = _write_quality_failure_frame(tmp_path, archive)

    assert evidence["output_frame_index"] == finite_frame.index
    assert evidence["output_frame_role"] == "last_finite_quality_failure_frame"
    assert (tmp_path / "optimized.mol2").is_file()


def test_trajectory_payload_counts_rejected_preliminary_attempts() -> None:
    class _SizedMain:
        coordinate_revision_count = 2
        topology_revision_count = 1
        selected_index = 1
        start = SimpleNamespace(value="ligand_build")
        frames = (
            SimpleNamespace(
                stage=SimpleNamespace(value="ligand_build"),
                event=SimpleNamespace(value="initial"),
            ),
        )

        def __len__(self):
            return 1

    archive = SimpleNamespace(
        main=_SizedMain(),
        ligand_build_attempts=(),
        preliminary_attempts=((1, 2),),
    )

    payload = _trajectory_payload(archive)

    assert payload["preliminary_attempt_count"] == 1
    assert payload["preliminary_attempt_frame_counts"] == [2]


def test_hotpot_auto_forcefield_error_persists_last_finite_frame(
    tmp_path: Path,
    monkeypatch,
) -> None:
    from hotpot import read_mol
    from hotpot.cheminfo import forcefields as ff
    from . import pipeline

    complex_mol = read_mol("[Am]N", fmt="smi")
    complex_mol.coordinates = np.asarray(
        ((0.0, 0.0, 0.0), (2.2, 0.0, 0.0)),
        dtype=float,
    )
    trajectory = ff.ForceFieldTrajectory.from_molecule(
        complex_mol,
        start=ff.TrajectoryStart.LIGAND_BUILD,
    )
    frame = trajectory.record_molecule(
        complex_mol,
        stage=ff.TrajectoryStage.LIGAND_BUILD,
        event=ff.TrajectoryEvent.INITIAL,
    )
    trajectory.select(frame.index)
    trajectory.set_terminal(frame.index)
    archive = ff.ForceFieldTrajectoryArchive(main=trajectory)
    error = ff.ForceFieldError("synthetic force-field failure")
    error.trajectory = archive
    cbond_result = SimpleNamespace(
        molecule=complex_mol,
        donor_indices=(1,),
        path_probability=1.0,
        steps=(),
    )

    monkeypatch.setattr(
        pipeline,
        "_infer_cbond",
        lambda ligand, metal, settings: cbond_result,
    )
    monkeypatch.setattr(
        ff,
        "auto_optimize",
        lambda *args, **kwargs: (_ for _ in ()).throw(error),
    )

    record = run_hotpot_auto_case(
        1,
        "N",
        str(tmp_path),
        "Am",
        SMOKE_SETTINGS,
        False,
    )

    assert record["status"] == "failed_forcefield"
    assert record["output_frame_role"] == "last_finite_failure_frame"
    assert (tmp_path / "cases/0001/optimized.mol2").is_file()
    assert (tmp_path / "cases/0001/optimized.sdf").is_file()
    assert (tmp_path / "cases/0001/trajectory/archive.json").is_file()


def test_cbond_inference_receives_both_recorded_thresholds(monkeypatch) -> None:
    from hotpot.cheminfo.AImodels.cbond import apply

    captured = {}
    expected_result = object()

    def fake_runtime(device):
        return f"runtime:{device}"

    def fake_auto_build_cbond(ligand, metal, **kwargs):
        captured.update(kwargs)
        return expected_result

    monkeypatch.setattr(apply, "get_cbond_runtime", fake_runtime)
    monkeypatch.setattr(apply, "auto_build_cbond", fake_auto_build_cbond)

    result = _infer_cbond(object(), "Eu", STANDARD_SETTINGS)

    assert result is expected_result
    assert captured == {
        "threshold": -0.125,
        "first_threshold": -0.5,
        "runtime": "runtime:cpu",
        "return_details": True,
    }


def test_cli_parses_case_selection_without_running_chemistry() -> None:
    arguments = build_parser().parse_args(
        ("--cases", "54,61,109", "--render", "required")
    )

    assert arguments.cases == (54, 61, 109)
    assert arguments.render == "required"


def test_optimizer_comparison_cli_declares_independent_workflow() -> None:
    arguments = build_optimizer_comparison_parser().parse_args(
        ("--cases", "1,187", "--workers", "2")
    )

    assert OPTIMIZER_COMPARISON_WORKFLOW == "shared-build-optimizer-comparison"
    assert arguments.cases == (1, 187)
    assert arguments.workers == 2


def test_optimizer_comparison_reports_paired_results_and_failures(
    tmp_path: Path,
) -> None:
    suite = BUILTIN_SUITES["extractants-eu-187"]
    failed_check = {
        "name": "short_bond",
        "passed": False,
        "severity": "error",
        "measured": 0.5,
        "threshold": 0.65,
        "atom_indices": [1, 2],
        "bond_indices": [3],
        "message": "Bond is too short",
    }
    records = [
        {
            "index": 1,
            "smiles": "N",
            "status": "compared",
            "cbond": {"donor_count": 1},
            "shared_build": {},
            "starting_point": {"verified_identical": True},
            "arms": {
                "complex_optimizer": {
                    "outcome": "passed",
                    "quality_passed": True,
                    "converged": True,
                    "elapsed_seconds": 2.0,
                    "optimization": {"final_energy": 10.0},
                },
                "ordinary_optimizer": {
                    "outcome": "failed_quality",
                    "quality_passed": False,
                    "converged": False,
                    "elapsed_seconds": 1.0,
                    "optimization": {"final_energy": 12.5},
                    "validation": {
                        "passed": False,
                        "checks": [failed_check],
                        "failures": [failed_check],
                    },
                },
            },
            "total_seconds": 4.0,
        },
        {
            "index": 2,
            "smiles": "O",
            "status": "failed_cbond",
            "error_type": "ValueError",
            "error_message": "no donor site",
            "total_seconds": 0.1,
        },
        {
            "index": 3,
            "smiles": "P",
            "status": "failed_optimizer",
            "cbond": {"donor_count": 1},
            "shared_build": {},
            "starting_point": {"verified_identical": True},
            "arms": {
                "complex_optimizer": {
                    "outcome": "failed_execution",
                    "quality_passed": None,
                    "converged": None,
                    "elapsed_seconds": 0.5,
                    "error_type": "RuntimeError",
                    "error_message": "backend stopped",
                    "optimization": {"final_energy": 99.0},
                },
                "ordinary_optimizer": {
                    "outcome": "passed",
                    "quality_passed": True,
                    "converged": True,
                    "elapsed_seconds": 0.75,
                    "optimization": {"final_energy": 12.0},
                },
            },
            "total_seconds": 2.0,
        },
    ]

    summary = aggregate_optimizer_comparison(
        tmp_path,
        records,
        suite,
        STANDARD_SETTINGS,
        RunProfile.STANDARD,
        (1, 2, 3),
        wall_seconds=4.1,
    )

    assert summary["cbond_failure_cases"] == [
        {
            "index": 2,
            "smiles": "O",
            "error_type": "ValueError",
            "error_message": "no donor site",
        }
    ]
    assert summary["paired"]["outcomes"] == {
        "pair_counts": {
            "passed | failed_quality": 1,
            "failed_execution | passed": 1,
        },
        "agreement_count": 0,
        "agreement_case_ids": [],
        "discordant_count": 2,
        "discordant_case_ids": [1, 3],
    }
    assert summary["paired"]["convergence"]["discordant_case_ids"] == [1, 3]
    assert summary["paired"]["convergence"]["pair_counts"][
        "unknown | true"
    ] == 1
    assert summary["arms"]["complex_optimizer"]["quality_unknown"] == 1
    assert summary["arms"]["complex_optimizer"]["convergence_unknown"] == 1
    assert summary["paired"]["timing"][
        "ordinary_minus_complex_aggregate_seconds"
    ] == -1.0
    assert summary["paired"]["final_energy_kj_mol"] == {
        "finite_pair_count": 1,
        "exact_equal_count": 0,
        "near_equal_tolerance": 1e-9,
        "near_equal_count": 0,
        "median_absolute_difference": 2.5,
        "maximum_absolute_difference": 2.5,
        "difference_over_1_kj_mol_count": 1,
        "top_divergent_cases": [{"index": 1, "absolute_difference": 2.5}],
    }
    assert summary["arm_failure_cases"]["ordinary_optimizer"][0][
        "validation_checks"
    ] == [failed_check]

    with (tmp_path / "comparison.csv").open(
        encoding="utf-8", newline=""
    ) as stream:
        rows = list(csv.DictReader(stream))
    assert rows[0]["ordinary_failure_details"]
    assert rows[0]["complex_failure_details"] == ""
    assert rows[1]["cbond_failure_reason"] == "ValueError: no donor site"
    assert rows[2]["absolute_final_energy_difference_kj_mol"] == ""
    report = (tmp_path / "report.md").read_text(encoding="utf-8")
    assert "## Paired outcomes" in report
    assert "Outcome disagreements (2): 0001, 0003" in report
    assert "## Paired timing" in report
    assert "## Paired final energy" in report
    assert "### CBond failures" in report
    assert "short_bond (atoms=[1, 2], bonds=[3]): Bond is too short" in report
    assert "full complex graph" in report
    assert "ligand-skeleton rings" in report


def test_aggregate_writes_reports_and_integrity(tmp_path: Path) -> None:
    suite = BUILTIN_SUITES["extractants-eu-187"]
    case_dir = tmp_path / "cases" / "0001"
    trajectory_dir = case_dir / "trajectory"
    trajectory_dir.mkdir(parents=True)
    (case_dir / "optimized.sdf").write_text("synthetic\n", encoding="utf-8")
    (case_dir / "optimized.mol2").write_text("synthetic\n", encoding="utf-8")
    (trajectory_dir / "archive.json").write_text("{}\n", encoding="utf-8")
    record = {
        "index": 1,
        "smiles": "N",
        "backend": "hotpot",
        "workflow": "cbond-complexes-build",
        "status": "passed",
        "phase": "complete",
        "cbond": {"donor_count": 1},
        "forcefield": {"effective_forcefield": "UFF"},
        "validation": {"passed": True, "failures": [], "warnings": []},
        "optimization": {
            "converged": True,
            "termination_reason": "converged",
            "final_energy": 1.0,
            "best_energy": 1.0,
        },
        "trajectory": {
            "main_frame_count": 2,
            "ligand_build_attempt_count": 1,
        },
        "output_frame_index": 1,
        "output_frame_role": "selected_success_frame",
        "visualization_topology_lossy": False,
        "cbond_seconds": 0.1,
        "ligand_build_seconds": 0.03,
        "coordination_restoration_seconds": 0.04,
        "complex_optimization_seconds": 0.12,
        "forcefield_seconds": 0.2,
        "total_seconds": 0.3,
    }
    (case_dir / "report.json").write_text(
        json.dumps(record) + "\n",
        encoding="utf-8",
    )

    summary = aggregate_run(
        tmp_path,
        [record],
        suite,
        STANDARD_SETTINGS,
        RunProfile.STANDARD,
        (1,),
        backend=BUILTIN_BACKENDS["hotpot"],
        wall_seconds=0.4,
        wall_seconds_scope="unit_test",
    )

    assert summary["overall_success_rate"] == 1.0
    assert summary["aggregate_ligand_build_seconds"] == 0.03
    assert summary["aggregate_coordination_restoration_seconds"] == 0.04
    assert summary["aggregate_complex_optimization_seconds"] == 0.12
    assert summary["workflow"] == "cbond-complexes-build"
    assert summary["cbond_success_count"] == 1
    assert summary["cbond_success_cases"] == [{"index": 1, "smiles": "N"}]
    assert (tmp_path / "results.csv").is_file()
    with (tmp_path / "results.csv").open(encoding="utf-8", newline="") as stream:
        result_row = next(csv.DictReader(stream))
    assert result_row["ligand_build_seconds"] == "0.03"
    assert result_row["coordination_restoration_seconds"] == "0.04"
    assert result_row["complex_optimization_seconds"] == "0.12"
    report_text = (tmp_path / "report.md").read_text(encoding="utf-8")
    assert "Aggregate Stage 1 ligand-build time: 0.030 s" in report_text
    assert (
        "Aggregate Stage 2 coordination-restoration time: 0.040 s"
        in report_text
    )
    assert (
        "Aggregate Stage 3 complex-optimization time: 0.120 s"
        in report_text
    )
    assert "## CBond-successful cases" in report_text
    assert "| 0001 | `N` |" in report_text
    assert "First CBond threshold: `-0.5`" in report_text
    assert "Subsequent CBond threshold: `-0.125`" in report_text
    assert (tmp_path / "summary.json").is_file()
    assert (tmp_path / "integrity.json").is_file()
    assert (tmp_path / "report.md").is_file()
    integrity = json.loads((tmp_path / "integrity.json").read_text())
    assert integrity["trajectory_archive_count"] == 1


def test_render_off_is_recorded_without_pymol_execution(tmp_path: Path) -> None:
    (tmp_path / "cases").mkdir()

    rendered = render_experiment(
        tmp_path,
        title="Synthetic",
        workers=1,
        mode="off",
    )

    assert rendered == []
    report = json.loads((tmp_path / "render_report.json").read_text())
    assert report["status"] == "disabled"


def test_existing_output_requires_explicit_resume(tmp_path: Path) -> None:
    configuration = {"suite": "synthetic"}
    _write_or_check_manifest(
        tmp_path,
        configuration,
        workers=1,
        resume=False,
    )

    try:
        _write_or_check_manifest(
            tmp_path,
            configuration,
            workers=1,
            resume=False,
        )
    except FileExistsError:
        pass
    else:
        raise AssertionError("an existing run must require --resume")


def test_starting_point_fingerprint_includes_coordinates() -> None:
    first = _SyntheticMol(np.asarray([[0.0, 0.0, 0.0]], dtype=float))
    second = _SyntheticMol(np.asarray([[1.0, 0.0, 0.0]], dtype=float))

    assert _starting_point_fingerprint(first) != _starting_point_fingerprint(second)


def test_optimizer_start_clone_preserves_molecule_metadata() -> None:
    from hotpot import read_mol

    molecule = read_mol("N")
    molecule.charge = 3
    molecule.properties = {"origin": {"case": 7}}

    clone_mol = _clone_starting_mol(molecule)

    assert clone_mol is not molecule
    assert clone_mol.charge == molecule.charge
    assert clone_mol.properties == molecule.properties
    assert clone_mol.properties is not molecule.properties
    assert _starting_point_fingerprint(clone_mol) == _starting_point_fingerprint(
        molecule
    )


def test_ordinary_optimizer_arm_calls_only_explicit_optimize(
    tmp_path: Path,
    monkeypatch,
) -> None:
    from hotpot.cheminfo import forcefields as ff

    calls = []

    def fake_optimize(mol, **kwargs):
        calls.append(("optimize", kwargs))
        return _SyntheticOptimizationReport()

    def forbidden_optimize_complex(mol, **kwargs):
        raise AssertionError("ordinary arm must not call optimize_complex")

    monkeypatch.setattr(ff, "optimize", fake_optimize)
    monkeypatch.setattr(ff, "optimize_complex", forbidden_optimize_complex)
    record = _run_optimizer_arm(
        "ordinary_optimizer",
        _SyntheticMol(np.asarray([[0.0, 0.0, 0.0]], dtype=float)),
        tmp_path,
        SMOKE_SETTINGS,
        seed=7,
    )

    assert record["outcome"] == "passed"
    assert calls[0][0] == "optimize"
    assert calls[0][1]["forcefield"] == "UFF"
    assert calls[0][1]["add_hydrogens"] is False
    assert calls[0][1]["quality_level"] == "standard"
    assert calls[0][1]["trajectory_path"] == tmp_path / "trajectory"


def test_failed_optimizer_execution_preserves_unknown_results(
    tmp_path: Path,
    monkeypatch,
) -> None:
    from hotpot.cheminfo import forcefields as ff

    def failed_optimize(mol, **kwargs):
        raise RuntimeError("backend stopped")

    monkeypatch.setattr(ff, "optimize", failed_optimize)
    record = _run_optimizer_arm(
        "ordinary_optimizer",
        _SyntheticMol(np.asarray([[0.0, 0.0, 0.0]], dtype=float)),
        tmp_path,
        SMOKE_SETTINGS,
        seed=7,
    )

    assert record["outcome"] == "failed_execution"
    assert record["quality_passed"] is None
    assert record["converged"] is None


def test_optimizer_comparison_builds_once_and_dispatches_identical_clones(
    tmp_path: Path,
    monkeypatch,
) -> None:
    import hotpot
    from hotpot.cheminfo import forcefields as ff
    from . import optimizer_comparison

    starting_mol = _SyntheticMol(
        np.asarray([[0.25, -0.5, 1.5]], dtype=float)
    )
    cbond_result = SimpleNamespace(
        molecule=starting_mol,
        donor_indices=(0,),
        path_probability=1.0,
        steps=(),
    )
    build_calls = []
    optimizer_calls = []

    monkeypatch.setattr(hotpot, "read_mol", lambda smiles, fmt: starting_mol)
    monkeypatch.setattr(
        optimizer_comparison,
        "_infer_cbond",
        lambda ligand, metal, settings: cbond_result,
    )

    def fake_build_complex3d(mol, forcefield, **kwargs):
        build_calls.append((mol, forcefield, kwargs))
        return _SyntheticBuildReport()

    def fake_optimize_complex(mol, **kwargs):
        optimizer_calls.append(
            ("optimize_complex", _starting_point_fingerprint(mol), kwargs)
        )
        return _SyntheticOptimizationReport()

    def fake_optimize(mol, **kwargs):
        optimizer_calls.append(("optimize", _starting_point_fingerprint(mol), kwargs))
        return _SyntheticOptimizationReport()

    monkeypatch.setattr(ff, "build_complex3d", fake_build_complex3d)
    monkeypatch.setattr(ff, "optimize_complex", fake_optimize_complex)
    monkeypatch.setattr(ff, "optimize", fake_optimize)

    record = run_optimizer_comparison_case(
        1,
        "N",
        str(tmp_path),
        "Eu",
        SMOKE_SETTINGS,
        resume=False,
    )

    assert len(build_calls) == 1
    assert [name for name, _, _ in optimizer_calls] == [
        "optimize_complex",
        "optimize",
    ]
    assert optimizer_calls[0][1] == optimizer_calls[1][1]
    assert record["starting_point"]["verified_identical"] is True
    assert record["arms"]["ordinary_optimizer"]["optimizer_api"] == (
        "forcefields.optimize"
    )
    assert record["status"] == "compared"


class _SyntheticMol:
    def __init__(self, coordinates: np.ndarray):
        self.atoms = [SimpleNamespace(id=0, atomic_number=7, formal_charge=0)]
        self.bonds = []
        self.coordinates = coordinates
        self.charge = 0
        self.properties = {}
        self.smiles = "N"

    def copy(self):
        clone_mol = _SyntheticMol(self.coordinates.copy())
        clone_mol.charge = self.charge
        clone_mol.properties = dict(self.properties)
        clone_mol.smiles = self.smiles
        return clone_mol

    def write(self, path: Path, *, overwrite: bool, write_single: bool) -> None:
        path.write_text("synthetic\n", encoding="utf-8")


@dataclass(frozen=True)
class _SyntheticQualityReport:
    level: str = "standard"
    passed: bool = True
    checks: tuple = ()
    failures: tuple = ()
    warnings: tuple = ()
    metrics: tuple = ()


@dataclass(frozen=True)
class _SyntheticOptimizationReport:
    quality_report: _SyntheticQualityReport = _SyntheticQualityReport()
    converged: bool = True
    termination_reason: str = "converged"
    trajectory: object = None


@dataclass(frozen=True)
class _SyntheticBuildReport:
    quality_report: _SyntheticQualityReport = _SyntheticQualityReport()
    trajectory: object = None
