"""Compare complex-aware and ordinary optimization from one built structure.

Each CBond-successful case calls :func:`forcefields.build_complex3d` exactly
once.  Two molecular copies are then optimized from that byte-identical
starting point by the explicit ``optimize_complex`` and ``optimize`` APIs.
"""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import multiprocessing as mp
import os
import traceback
import warnings
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from time import perf_counter
from typing import Mapping, Optional, Sequence, TYPE_CHECKING

import numpy as np

from .configuration import (
    BUILTIN_SUITES,
    REPOSITORY_ROOT,
    BenchmarkSettings,
    BenchmarkSuite,
    RunProfile,
)
from .io import json_value, load_case_reports, load_smiles, sha256_file, write_json
from .pipeline import (
    _cbond_payload,
    _infer_cbond,
    _quality_payload,
    _trajectory_payload,
    _write_last_finite_failure_frame,
)
from .runner import (
    _write_or_check_manifest,
    resolve_suite,
    settings_with_overrides,
)

if TYPE_CHECKING:
    from hotpot import Molecule


WORKFLOW = "shared-build-optimizer-comparison"
ARM_NAMES = ("complex_optimizer", "ordinary_optimizer")
DEFAULT_OUTPUT = (
    REPOSITORY_ROOT / "movie/benchmarks/extractants_eu_187_optimizer_comparison"
)


def _starting_point_fingerprint(mol: "Molecule") -> str:
    """Return an exact atom, bond, and coordinate identity for one structure."""
    atoms = [
        (int(atom.id), int(atom.atomic_number), int(atom.formal_charge))
        for atom in mol.atoms
    ]
    bonds = sorted(
        (
            min(int(bond.atom1.idx), int(bond.atom2.idx)),
            max(int(bond.atom1.idx), int(bond.atom2.idx)),
            float(bond.bond_order),
            bond.bond_kind.value,
        )
        for bond in mol.bonds
    )
    coordinates = np.ascontiguousarray(mol.coordinates, dtype="<f8")
    digest = hashlib.sha256()
    digest.update(
        json.dumps(
            {"charge": int(mol.charge), "atoms": atoms, "bonds": bonds},
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    )
    digest.update(str(coordinates.shape).encode("ascii"))
    digest.update(coordinates.tobytes())
    return digest.hexdigest()


def _clone_starting_mol(mol: "Molecule") -> "Molecule":
    """Clone a comparison start while retaining molecule-level metadata."""
    clone_mol = mol.copy()
    clone_mol.charge = mol.charge
    clone_mol.properties = copy.deepcopy(mol.properties)
    return clone_mol


def _write_structure(directory: Path, mol: "Molecule", stem: str) -> None:
    mol.write(directory / f"{stem}.mol2", overwrite=True, write_single=True)
    mol.write(directory / f"{stem}.sdf", overwrite=True, write_single=True)


def _run_optimizer_arm(
    arm_name: str,
    starting_mol: "Molecule",
    arm_dir: Path,
    settings: BenchmarkSettings,
    seed: int,
) -> dict[str, object]:
    """Run one explicit optimizer and persist its independent evidence."""
    from hotpot.cheminfo import forcefields as ff

    arm_dir.mkdir(parents=True, exist_ok=True)
    started = perf_counter()
    record: dict[str, object] = {
        "arm": arm_name,
        "optimizer_api": (
            "forcefields.optimize_complex"
            if arm_name == "complex_optimizer"
            else "forcefields.optimize"
        ),
        "outcome": "failed_execution",
        "starting_point_fingerprint": _starting_point_fingerprint(starting_mol),
        "settings": {
            "forcefield": "UFF",
            "epochs": settings.epochs,
            "steps_per_epoch": settings.steps_per_epoch,
            "add_hydrogens": False,
            "quality_level": settings.quality_level,
            "seed": seed,
            "perturb_sigma": settings.perturb_sigma,
        },
    }
    optimizer_kwargs = {
        "forcefield": "UFF",
        "epochs": settings.epochs,
        "steps_per_epoch": settings.steps_per_epoch,
        "add_hydrogens": False,
        "quality_level": settings.quality_level,
        "seed": seed,
        "perturb_sigma": settings.perturb_sigma,
        "save_movie": True,
        "trajectory_path": arm_dir / "trajectory",
    }
    if arm_name == "complex_optimizer":
        optimizer = ff.optimize_complex
        optimizer_kwargs["complex_untangling_attempts"] = (
            settings.complex_untangling_attempts
        )
        optimizer_kwargs["trajectory_start"] = ff.TrajectoryStart.COMPLEX_UNTANGLING
    else:
        optimizer = ff.optimize
        optimizer_kwargs["trajectory_start"] = ff.TrajectoryStart.FINAL_OPTIMIZATION

    emitted_messages: list[str] = []
    try:
        with warnings.catch_warnings(record=True) as emitted_warnings:
            warnings.simplefilter("always")
            optimization_report = optimizer(starting_mol, **optimizer_kwargs)
        emitted_messages = [str(item.message) for item in emitted_warnings]
    except ff.ForceFieldError as error:
        emitted_messages.extend(
            str(item.message) for item in locals().get("emitted_warnings", ())
        )
        record.update(
            error_type=type(error).__name__,
            error_message=str(error),
            traceback=traceback.format_exc(),
            warnings=emitted_messages,
            trajectory=_trajectory_payload(error.trajectory),
        )
        error_report = getattr(error, "report", None)
        if isinstance(error_report, ff.ForceFieldValidationReport):
            record["validation"] = _quality_payload(error_report)
        elif isinstance(error_report, ff.ForceFieldRunReport):
            record["optimization"] = json_value(error_report)
        if error.trajectory is not None:
            try:
                record.update(
                    _write_last_finite_failure_frame(arm_dir, error.trajectory)
                )
            except Exception as output_error:
                record["output_structure_unavailable_reason"] = (
                    f"{type(output_error).__name__}: {output_error}"
                )
    except Exception as error:
        record.update(
            error_type=type(error).__name__,
            error_message=str(error),
            traceback=traceback.format_exc(),
            warnings=emitted_messages,
        )
    else:
        validation = _quality_payload(optimization_report.quality_report)
        record.update(
            outcome=("passed" if validation["passed"] else "failed_quality"),
            quality_passed=bool(validation["passed"]),
            converged=bool(optimization_report.converged),
            termination_reason=optimization_report.termination_reason,
            optimization=json_value(optimization_report),
            validation=validation,
            warnings=emitted_messages,
            trajectory=_trajectory_payload(optimization_report.trajectory),
        )
        _write_structure(arm_dir, starting_mol, "optimized")

    validation = record.get("validation") or {}
    optimization = record.get("optimization") or {}
    record["quality_passed"] = (
        bool(validation["passed"]) if "passed" in validation else None
    )
    if "converged" in record:
        record["converged"] = bool(record["converged"])
    else:
        record["converged"] = (
            bool(optimization["converged"])
            if "converged" in optimization
            else None
        )
    record["elapsed_seconds"] = perf_counter() - started
    write_json(arm_dir / "report.json", record)
    return record


def run_optimizer_comparison_case(
    index: int,
    smiles: str,
    output_root_text: str,
    metal: str,
    settings: BenchmarkSettings,
    resume: bool,
) -> dict[str, object]:
    """Run CBond, one shared complex build, and two explicit optimizer arms."""
    from hotpot import read_mol
    from hotpot.cheminfo import forcefields as ff

    case_dir = Path(output_root_text) / "cases" / f"{index:04d}"
    case_dir.mkdir(parents=True, exist_ok=True)
    report_path = case_dir / "report.json"
    if resume and report_path.is_file():
        return json.loads(report_path.read_text(encoding="utf-8"))

    seed = settings.seed + index
    record: dict[str, object] = {
        "index": index,
        "smiles": smiles,
        "metal": metal,
        "backend": "hotpot",
        "workflow": WORKFLOW,
        "status": "running",
        "phase": "read",
        "settings": settings.to_manifest(),
    }
    (case_dir / "input.smi").write_text(smiles + "\n", encoding="utf-8")
    started = perf_counter()

    try:
        ligand = read_mol(smiles, fmt="smi")
        record["input_atom_count"] = len(ligand.atoms)

        record["phase"] = "cbond"
        phase_started = perf_counter()
        cbond_result = _infer_cbond(ligand, metal, settings)
        common_mol = cbond_result.molecule
        record["cbond_seconds"] = perf_counter() - phase_started
        record["cbond"] = _cbond_payload(cbond_result)
        (case_dir / "cbond.smi").write_text(
            common_mol.smiles + "\n",
            encoding="utf-8",
        )

        record["phase"] = "shared_build"
        build_dir = case_dir / "shared_build"
        build_dir.mkdir(exist_ok=True)
        phase_started = perf_counter()
        build_report = ff.build_complex3d(
            common_mol,
            "UFF",
            max_attempts=settings.max_attempts,
            candidate_warmup_steps=settings.candidate_warmup_steps,
            candidate_score_steps=settings.candidate_score_steps,
            best_candidate_refine_steps=settings.best_candidate_refine_steps,
            ligand_untangling_attempts=settings.ligand_untangling_attempts,
            coordination_restoration_attempts=(
                settings.coordination_restoration_attempts
            ),
            coordination_relaxation_steps=settings.coordination_relaxation_steps,
            timeout=settings.timeout,
            add_hydrogens=True,
            seed=seed,
            perturb_sigma=settings.perturb_sigma,
            save_movie=True,
            trajectory_start=ff.TrajectoryStart.COORDINATION_RESTORATION,
            trajectory_path=build_dir / "trajectory",
        )
        record["shared_build_seconds"] = perf_counter() - phase_started
        record["shared_build"] = {
            "report": json_value(build_report),
            "validation": _quality_payload(build_report.quality_report),
            "trajectory": _trajectory_payload(build_report.trajectory),
        }
        _write_structure(build_dir, common_mol, "starting_structure")

        common_fingerprint = _starting_point_fingerprint(common_mol)
        complex_optimizer_mol = _clone_starting_mol(common_mol)
        ordinary_optimizer_mol = _clone_starting_mol(common_mol)
        arm_fingerprints = {
            "complex_optimizer": _starting_point_fingerprint(complex_optimizer_mol),
            "ordinary_optimizer": _starting_point_fingerprint(
                ordinary_optimizer_mol
            ),
        }
        starting_point_verified = all(
            fingerprint == common_fingerprint
            for fingerprint in arm_fingerprints.values()
        )
        record["starting_point"] = {
            "fingerprint": common_fingerprint,
            "arm_fingerprints": arm_fingerprints,
            "verified_identical": starting_point_verified,
            "atom_count": len(common_mol.atoms),
            "bond_count": len(common_mol.bonds),
        }
        if not starting_point_verified:
            raise RuntimeError("optimizer arms do not share an identical start")

        record["phase"] = "optimization"
        arms = {
            "complex_optimizer": _run_optimizer_arm(
                "complex_optimizer",
                complex_optimizer_mol,
                case_dir / "complex_optimizer",
                settings,
                seed,
            ),
            "ordinary_optimizer": _run_optimizer_arm(
                "ordinary_optimizer",
                ordinary_optimizer_mol,
                case_dir / "ordinary_optimizer",
                settings,
                seed,
            ),
        }
        record["arms"] = arms
        record["status"] = (
            "compared"
            if all(arm["outcome"] != "failed_execution" for arm in arms.values())
            else "failed_optimizer"
        )
        record["phase"] = "complete"
    except ValueError as error:
        record.update(
            status=("failed_cbond" if record["phase"] == "cbond" else "failed_build"),
            error_type=type(error).__name__,
            error_message=str(error),
            traceback=traceback.format_exc(),
        )
    except ff.ForceFieldError as error:
        record.update(
            status="failed_build",
            error_type=type(error).__name__,
            error_message=str(error),
            traceback=traceback.format_exc(),
            build_failure_trajectory=_trajectory_payload(error.trajectory),
        )
        if error.trajectory is not None:
            build_dir = case_dir / "shared_build"
            try:
                record.update(
                    _write_last_finite_failure_frame(build_dir, error.trajectory)
                )
            except Exception as output_error:
                record["output_structure_unavailable_reason"] = (
                    f"{type(output_error).__name__}: {output_error}"
                )
    except Exception as error:
        record.update(
            status="failed_internal",
            error_type=type(error).__name__,
            error_message=str(error),
            traceback=traceback.format_exc(),
        )

    record["total_seconds"] = perf_counter() - started
    write_json(report_path, record)
    return record


def _comparison_row(record: Mapping[str, object]) -> dict[str, object]:
    arms = record.get("arms") or {}
    complex_arm = arms.get("complex_optimizer") or {}
    ordinary_arm = arms.get("ordinary_optimizer") or {}
    return {
        "index": record["index"],
        "smiles": record["smiles"],
        "case_status": record["status"],
        "cbond_succeeded": bool(record.get("cbond")),
        "cbond_failure_reason": _case_failure_reason(record),
        "shared_start_verified": bool(
            (record.get("starting_point") or {}).get("verified_identical", False)
        ),
        "complex_outcome": complex_arm.get("outcome"),
        "complex_quality_passed": complex_arm.get("quality_passed"),
        "complex_converged": complex_arm.get("converged"),
        "complex_seconds": complex_arm.get("elapsed_seconds"),
        "complex_final_energy_kj_mol": _finite_final_energy(complex_arm),
        "complex_failure_details": _arm_failure_details(complex_arm),
        "ordinary_outcome": ordinary_arm.get("outcome"),
        "ordinary_quality_passed": ordinary_arm.get("quality_passed"),
        "ordinary_converged": ordinary_arm.get("converged"),
        "ordinary_seconds": ordinary_arm.get("elapsed_seconds"),
        "ordinary_final_energy_kj_mol": _finite_final_energy(ordinary_arm),
        "absolute_final_energy_difference_kj_mol": _paired_energy_difference(
            complex_arm, ordinary_arm
        ),
        "ordinary_failure_details": _arm_failure_details(ordinary_arm),
        "total_seconds": record.get("total_seconds"),
    }


def _case_failure_reason(record: Mapping[str, object]) -> str:
    if record.get("status") != "failed_cbond":
        return ""
    error_type = str(record.get("error_type") or "")
    error_message = str(record.get("error_message") or "")
    return ": ".join(part for part in (error_type, error_message) if part)


def _failed_validation_checks(
    arm: Mapping[str, object],
) -> list[Mapping[str, object]]:
    validation = arm.get("validation") or {}
    failures = validation.get("failures") or []
    if failures:
        return list(failures)
    return [
        check
        for check in validation.get("checks") or []
        if not bool(check.get("passed")) and check.get("severity") == "error"
    ]


def _arm_failure_details(arm: Mapping[str, object]) -> str:
    if not arm or arm.get("outcome") == "passed":
        return ""
    details = {
        "error_type": arm.get("error_type"),
        "error_message": arm.get("error_message"),
        "validation_checks": _failed_validation_checks(arm),
    }
    return json.dumps(
        {key: value for key, value in details.items() if value},
        sort_keys=True,
        separators=(",", ":"),
    )


def _finite_final_energy(arm: Mapping[str, object]) -> Optional[float]:
    energy = (arm.get("optimization") or {}).get("final_energy")
    if energy is None:
        return None
    value = float(energy)
    return value if np.isfinite(value) else None


def _absolute_energy_difference(
    complex_arm: Mapping[str, object],
    ordinary_arm: Mapping[str, object],
) -> Optional[float]:
    complex_energy = _finite_final_energy(complex_arm)
    ordinary_energy = _finite_final_energy(ordinary_arm)
    if complex_energy is None or ordinary_energy is None:
        return None
    return abs(ordinary_energy - complex_energy)


def _paired_energy_difference(
    complex_arm: Mapping[str, object],
    ordinary_arm: Mapping[str, object],
) -> Optional[float]:
    if any(
        arm.get("outcome") == "failed_execution"
        for arm in (complex_arm, ordinary_arm)
    ):
        return None
    return _absolute_energy_difference(complex_arm, ordinary_arm)


def _state_label(value: object) -> str:
    if value is None:
        return "unknown"
    return str(bool(value)).lower()


def _paired_summary(
    records: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    paired = [
        record
        for record in records
        if all(name in (record.get("arms") or {}) for name in ARM_NAMES)
    ]

    outcome_pairs: Counter[str] = Counter()
    convergence_pairs: Counter[str] = Counter()
    outcome_agreement: list[int] = []
    outcome_discordant: list[int] = []
    convergence_agreement: list[int] = []
    convergence_discordant: list[int] = []
    complex_seconds: list[float] = []
    ordinary_seconds: list[float] = []
    complex_faster: list[int] = []
    ordinary_faster: list[int] = []
    equal_timing: list[int] = []
    energy_differences: list[tuple[int, float]] = []

    for record in paired:
        index = int(record["index"])
        arms = record["arms"]
        complex_arm = arms["complex_optimizer"]
        ordinary_arm = arms["ordinary_optimizer"]
        complex_outcome = str(complex_arm["outcome"])
        ordinary_outcome = str(ordinary_arm["outcome"])
        outcome_pairs[f"{complex_outcome} | {ordinary_outcome}"] += 1
        (outcome_agreement if complex_outcome == ordinary_outcome else outcome_discordant).append(index)

        complex_converged = complex_arm.get("converged")
        ordinary_converged = ordinary_arm.get("converged")
        convergence_pairs[
            f"{_state_label(complex_converged)} | {_state_label(ordinary_converged)}"
        ] += 1
        (
            convergence_agreement
            if complex_converged == ordinary_converged
            else convergence_discordant
        ).append(index)

        both_executed = all(
            arm.get("outcome") != "failed_execution"
            for arm in (complex_arm, ordinary_arm)
        )
        if both_executed:
            complex_elapsed = float(complex_arm["elapsed_seconds"])
            ordinary_elapsed = float(ordinary_arm["elapsed_seconds"])
            complex_seconds.append(complex_elapsed)
            ordinary_seconds.append(ordinary_elapsed)
            if complex_elapsed < ordinary_elapsed:
                complex_faster.append(index)
            elif ordinary_elapsed < complex_elapsed:
                ordinary_faster.append(index)
            else:
                equal_timing.append(index)
        energy_difference = _paired_energy_difference(complex_arm, ordinary_arm)
        if energy_difference is not None:
            energy_differences.append((index, energy_difference))

    complex_total = sum(complex_seconds)
    ordinary_total = sum(ordinary_seconds)
    deltas = [
        ordinary - complex
        for complex, ordinary in zip(complex_seconds, ordinary_seconds)
    ]
    absolute_energy_differences = [difference for _, difference in energy_differences]
    top_energy_divergences = sorted(
        energy_differences,
        key=lambda item: item[1],
        reverse=True,
    )[:10]
    return {
        "paired_case_count": len(paired),
        "outcomes": {
            "pair_counts": dict(outcome_pairs),
            "agreement_count": len(outcome_agreement),
            "agreement_case_ids": outcome_agreement,
            "discordant_count": len(outcome_discordant),
            "discordant_case_ids": outcome_discordant,
        },
        "convergence": {
            "pair_counts": dict(convergence_pairs),
            "agreement_count": len(convergence_agreement),
            "agreement_case_ids": convergence_agreement,
            "discordant_count": len(convergence_discordant),
            "discordant_case_ids": convergence_discordant,
        },
        "timing": {
            "completed_pair_count": len(complex_seconds),
            "complex_aggregate_seconds": complex_total,
            "ordinary_aggregate_seconds": ordinary_total,
            "ordinary_minus_complex_aggregate_seconds": ordinary_total - complex_total,
            "ordinary_to_complex_aggregate_ratio": (
                ordinary_total / complex_total if complex_total else None
            ),
            "ordinary_vs_complex_aggregate_percent": (
                100.0 * (ordinary_total - complex_total) / complex_total
                if complex_total
                else None
            ),
            "median_ordinary_minus_complex_seconds": (
                float(np.median(np.asarray(deltas, dtype=float))) if deltas else None
            ),
            "complex_faster_case_ids": complex_faster,
            "ordinary_faster_case_ids": ordinary_faster,
            "equal_timing_case_ids": equal_timing,
        },
        "final_energy_kj_mol": {
            "finite_pair_count": len(energy_differences),
            "exact_equal_count": sum(
                difference == 0.0 for difference in absolute_energy_differences
            ),
            "near_equal_tolerance": 1e-9,
            "near_equal_count": sum(
                difference <= 1e-9 for difference in absolute_energy_differences
            ),
            "median_absolute_difference": (
                float(np.median(np.asarray(absolute_energy_differences, dtype=float)))
                if absolute_energy_differences
                else None
            ),
            "maximum_absolute_difference": (
                max(absolute_energy_differences)
                if absolute_energy_differences
                else None
            ),
            "difference_over_1_kj_mol_count": sum(
                difference > 1.0 for difference in absolute_energy_differences
            ),
            "top_divergent_cases": [
                {"index": index, "absolute_difference": difference}
                for index, difference in top_energy_divergences
            ],
        },
    }


def _failure_summaries(
    records: Sequence[Mapping[str, object]],
) -> tuple[list[dict[str, object]], dict[str, list[dict[str, object]]]]:
    cbond_failures = [
        {
            "index": int(record["index"]),
            "smiles": str(record["smiles"]),
            "error_type": record.get("error_type"),
            "error_message": record.get("error_message"),
        }
        for record in records
        if record.get("status") == "failed_cbond"
    ]
    arm_failures: dict[str, list[dict[str, object]]] = {}
    for arm_name in ARM_NAMES:
        arm_failures[arm_name] = [
            {
                "index": int(record["index"]),
                "smiles": str(record["smiles"]),
                "outcome": arm["outcome"],
                "error_type": arm.get("error_type"),
                "error_message": arm.get("error_message"),
                "validation_checks": _failed_validation_checks(arm),
            }
            for record in records
            if (arm := (record.get("arms") or {}).get(arm_name))
            and arm.get("outcome") != "passed"
        ]
    return cbond_failures, arm_failures


def _format_case_ids(case_ids: Sequence[int]) -> str:
    if not case_ids:
        return "None"
    ordered = sorted(set(case_ids))
    ranges: list[str] = []
    start = previous = ordered[0]
    for case_id in ordered[1:]:
        if case_id == previous + 1:
            previous = case_id
            continue
        ranges.append(
            f"{start:04d}" if start == previous else f"{start:04d}-{previous:04d}"
        )
        start = previous = case_id
    ranges.append(
        f"{start:04d}" if start == previous else f"{start:04d}-{previous:04d}"
    )
    return ", ".join(ranges)


def _validation_check_text(check: Mapping[str, object]) -> str:
    locations = []
    if check.get("atom_indices"):
        locations.append(f"atoms={check['atom_indices']}")
    if check.get("bond_indices"):
        locations.append(f"bonds={check['bond_indices']}")
    location_text = f" ({', '.join(locations)})" if locations else ""
    return f"{check.get('name')}{location_text}: {check.get('message')}"


def _format_optional_float(value: Optional[float], digits: int = 6) -> str:
    return "not available" if value is None else f"{value:.{digits}f}"


def _arm_summary(
    records: Sequence[Mapping[str, object]],
    arm_name: str,
) -> dict[str, object]:
    arm_records = [
        record["arms"][arm_name]
        for record in records
        if arm_name in (record.get("arms") or {})
    ]
    elapsed = [float(record["elapsed_seconds"]) for record in arm_records]
    return {
        "attempted": len(arm_records),
        "outcomes": dict(Counter(str(record["outcome"]) for record in arm_records)),
        "quality_passed": sum(record.get("quality_passed") is True for record in arm_records),
        "quality_unknown": sum(record.get("quality_passed") is None for record in arm_records),
        "converged": sum(record.get("converged") is True for record in arm_records),
        "convergence_unknown": sum(record.get("converged") is None for record in arm_records),
        "aggregate_seconds": sum(elapsed),
        "median_seconds": (
            float(np.median(np.asarray(elapsed, dtype=float))) if elapsed else None
        ),
    }


def aggregate_optimizer_comparison(
    output_root: Path,
    records: Sequence[Mapping[str, object]],
    suite: BenchmarkSuite,
    settings: BenchmarkSettings,
    profile: RunProfile,
    expected_indices: Sequence[int],
    *,
    wall_seconds: Optional[float],
) -> dict[str, object]:
    """Write comparison tables and a concise human-readable report."""
    ordered_records = sorted(records, key=lambda item: int(item["index"]))
    rows = [_comparison_row(record) for record in ordered_records]
    if rows:
        with (output_root / "comparison.csv").open(
            "w", encoding="utf-8", newline=""
        ) as stream:
            writer = csv.DictWriter(stream, fieldnames=tuple(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

    cbond_success_cases = [
        {"index": int(record["index"]), "smiles": str(record["smiles"])}
        for record in ordered_records
        if record.get("cbond")
    ]
    cbond_failure_cases, arm_failure_cases = _failure_summaries(ordered_records)
    paired = _paired_summary(ordered_records)
    summary = {
        "workflow": WORKFLOW,
        "suite": suite.to_manifest(),
        "profile": profile.value,
        "settings": settings.to_manifest(),
        "sample_count": len(ordered_records),
        "expected_case_count": len(expected_indices),
        "missing_report_indices": sorted(
            set(expected_indices)
            - {int(record["index"]) for record in ordered_records}
        ),
        "case_status_counts": dict(
            Counter(str(record["status"]) for record in ordered_records)
        ),
        "cbond_success_count": len(cbond_success_cases),
        "cbond_success_cases": cbond_success_cases,
        "cbond_failure_count": len(cbond_failure_cases),
        "cbond_failure_cases": cbond_failure_cases,
        "shared_build_success_count": sum(
            record.get("shared_build") is not None for record in ordered_records
        ),
        "shared_build_trajectory_count": sum(
            (
                output_root
                / "cases"
                / f"{int(record['index']):04d}"
                / "shared_build"
                / "trajectory"
                / "archive.json"
            ).is_file()
            for record in ordered_records
        ),
        "identical_start_verified_count": sum(
            bool(
                (record.get("starting_point") or {}).get(
                    "verified_identical", False
                )
            )
            for record in ordered_records
        ),
        "arms": {
            arm_name: _arm_summary(ordered_records, arm_name)
            for arm_name in ARM_NAMES
        },
        "arm_failure_cases": arm_failure_cases,
        "paired": paired,
        "wall_seconds": wall_seconds,
    }
    for arm_name in ARM_NAMES:
        summary["arms"][arm_name]["trajectory_archive_count"] = sum(
            (
                output_root
                / "cases"
                / f"{int(record['index']):04d}"
                / arm_name
                / "trajectory"
                / "archive.json"
            ).is_file()
            for record in ordered_records
        )
    write_json(output_root / "summary.json", summary)

    cbond_rows = ["| Case | Ligand SMILES |", "|---:|---|"]
    for item in cbond_success_cases:
        escaped_smiles = item["smiles"].replace("|", "\\|")
        cbond_rows.append(f"| {item['index']:04d} | `{escaped_smiles}` |")
    if len(cbond_rows) == 2:
        cbond_rows.append("| - | None |")
    arm_rows = [
        "| Optimizer API | Attempted | Passed | Quality failed | Execution failed | Quality unknown | Converged | Convergence unknown | Seconds |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for arm_name in ARM_NAMES:
        arm = summary["arms"][arm_name]
        outcomes = arm["outcomes"]
        arm_rows.append(
            f"| `{arm_name}` | {arm['attempted']} | "
            f"{outcomes.get('passed', 0)} | {outcomes.get('failed_quality', 0)} | "
            f"{outcomes.get('failed_execution', 0)} | {arm['quality_unknown']} | "
            f"{arm['converged']} | {arm['convergence_unknown']} | "
            f"{arm['aggregate_seconds']:.3f} |"
        )
    outcome_rows = [
        "| Complex outcome | Ordinary outcome | Cases |",
        "|---|---|---:|",
    ]
    for pair, count in paired["outcomes"]["pair_counts"].items():
        complex_value, ordinary_value = pair.split(" | ", maxsplit=1)
        outcome_rows.append(f"| `{complex_value}` | `{ordinary_value}` | {count} |")
    convergence_rows = [
        "| Complex converged | Ordinary converged | Cases |",
        "|---|---|---:|",
    ]
    for pair, count in paired["convergence"]["pair_counts"].items():
        complex_value, ordinary_value = pair.split(" | ", maxsplit=1)
        convergence_rows.append(f"| `{complex_value}` | `{ordinary_value}` | {count} |")
    cbond_failure_rows = [
        "| Case | Ligand SMILES | Reason |",
        "|---:|---|---|",
    ]
    for failure in cbond_failure_cases:
        escaped_smiles = failure["smiles"].replace("|", "\\|")
        reason = _case_failure_reason(
            {"status": "failed_cbond", **failure}
        ).replace("|", "\\|")
        cbond_failure_rows.append(
            f"| {failure['index']:04d} | `{escaped_smiles}` | {reason} |"
        )
    if len(cbond_failure_rows) == 2:
        cbond_failure_rows.append("| - | None | - |")
    arm_failure_rows = [
        "| Optimizer | Case | Outcome | Failed validation checks / error |",
        "|---|---:|---|---|",
    ]
    for arm_name in ARM_NAMES:
        for failure in arm_failure_cases[arm_name]:
            checks = failure["validation_checks"]
            detail = "; ".join(
                _validation_check_text(check) for check in checks
            )
            if not detail:
                detail = ": ".join(
                    str(value)
                    for value in (
                        failure.get("error_type"),
                        failure.get("error_message"),
                    )
                    if value
                )
            escaped_detail = detail.replace("|", "\\|")
            arm_failure_rows.append(
                f"| `{arm_name}` | {failure['index']:04d} | "
                f"`{failure['outcome']}` | {escaped_detail} |"
            )
    if len(arm_failure_rows) == 2:
        arm_failure_rows.append("| - | - | None | - |")
    timing = paired["timing"]
    energy = paired["final_energy_kj_mol"]
    wall_time_text = (
        f"{float(summary['wall_seconds']):.3f} s"
        if summary["wall_seconds"] is not None
        else "not measured during aggregate-only reporting"
    )
    energy_divergence_rows = [
        "| Rank | Case | Absolute final-energy difference (kJ/mol) |",
        "|---:|---:|---:|",
    ]
    for rank, item in enumerate(energy["top_divergent_cases"], start=1):
        energy_divergence_rows.append(
            f"| {rank} | {item['index']:04d} | {item['absolute_difference']:.6f} |"
        )
    report = f"""# Shared-start optimizer comparison

## Protocol

- Workflow: `{WORKFLOW}`
- Suite: `{suite.name}`
- Profile: `{profile.value}`
- First CBond threshold: `{settings.first_cbond_threshold}`
- Subsequent CBond threshold: `{settings.subsequent_cbond_threshold}`
- Shared builder: `forcefields.build_complex3d(..., "UFF")`, called once per case
- Arm A: `forcefields.optimize_complex(..., "UFF")`
- Arm B: `forcefields.optimize(..., "UFF")`
- Both arms: `add_hydrogens=False`, identical epochs, steps, seed and quality level
- Starting-point identity: exact atom, bond and float64-coordinate SHA-256
- Final energy: each arm's reported terminal-frame UFF energy in kJ/mol
- Acceptance scope follows each public API: `optimize_complex` inspects the
  full complex graph, while ordinary `optimize` inspects ligand-skeleton rings.
  Equal pass counts therefore describe each API's operational contract, not
  evaluation by one identical ring scope.

The optimizer outcome is deliberately three-state: `passed`,
`failed_quality`, or `failed_execution`. Quality-gate passage and optimizer
convergence are reported independently.

## Results

{chr(10).join(arm_rows)}

- Cases: {summary['sample_count']}
- CBond successes: {summary['cbond_success_count']}
- Shared builds: {summary['shared_build_success_count']}
- Shared-build trajectories: {summary['shared_build_trajectory_count']}
- Identical starts verified: {summary['identical_start_verified_count']}
- Missing case reports: {summary['missing_report_indices']}
- Wall time: {wall_time_text}

## Paired outcomes

{chr(10).join(outcome_rows)}

- Outcome agreements ({paired['outcomes']['agreement_count']}): {_format_case_ids(paired['outcomes']['agreement_case_ids'])}
- Outcome disagreements ({paired['outcomes']['discordant_count']}): {_format_case_ids(paired['outcomes']['discordant_case_ids'])}

## Paired convergence

{chr(10).join(convergence_rows)}

- Convergence agreements ({paired['convergence']['agreement_count']}): {_format_case_ids(paired['convergence']['agreement_case_ids'])}
- Convergence disagreements ({paired['convergence']['discordant_count']}): {_format_case_ids(paired['convergence']['discordant_case_ids'])}

## Paired timing

- Completed optimizer pairs: {timing['completed_pair_count']}
- Complex-aware aggregate: {timing['complex_aggregate_seconds']:.3f} s
- Ordinary aggregate: {timing['ordinary_aggregate_seconds']:.3f} s
- Ordinary minus complex-aware aggregate: {timing['ordinary_minus_complex_aggregate_seconds']:.3f} s
- Ordinary / complex-aware aggregate ratio: {_format_optional_float(timing['ordinary_to_complex_aggregate_ratio'])}
- Ordinary relative to complex-aware aggregate: {_format_optional_float(timing['ordinary_vs_complex_aggregate_percent'], 3)}%
- Median paired ordinary-minus-complex difference: {_format_optional_float(timing['median_ordinary_minus_complex_seconds'])} s
- Complex-aware faster cases: {_format_case_ids(timing['complex_faster_case_ids'])}
- Ordinary faster cases: {_format_case_ids(timing['ordinary_faster_case_ids'])}
- Equal-time cases: {_format_case_ids(timing['equal_timing_case_ids'])}

## Paired final energy

- Finite-energy pairs: {energy['finite_pair_count']}
- Exactly equal: {energy['exact_equal_count']}
- Equal within {energy['near_equal_tolerance']} kJ/mol: {energy['near_equal_count']}
- Median absolute difference: {_format_optional_float(energy['median_absolute_difference'])} kJ/mol
- Maximum absolute difference: {_format_optional_float(energy['maximum_absolute_difference'])} kJ/mol
- Differences greater than 1 kJ/mol: {energy['difference_over_1_kj_mol_count']}

{chr(10).join(energy_divergence_rows)}

## Failures

### CBond failures

{chr(10).join(cbond_failure_rows)}

### Optimizer-arm failures

{chr(10).join(arm_failure_rows)}

## CBond-successful cases

{chr(10).join(cbond_rows)}

## Artifacts

- `manifest.json`: immutable workflow, corpus and settings identity
- `comparison.csv`: paired per-case outcomes
- `summary.json`: aggregate counts and timings
- `cases/NNNN/shared_build/trajectory/`: the one common construction trajectory
- `cases/NNNN/complex_optimizer/`: complex-aware report, trajectory and structure
- `cases/NNNN/ordinary_optimizer/`: ordinary report, trajectory and structure
"""
    (output_root / "report.md").write_text(report, encoding="utf-8")
    return summary


def run_optimizer_comparison(
    suite: BenchmarkSuite,
    output_root: Path,
    *,
    profile: RunProfile,
    settings: BenchmarkSettings,
    workers: int,
    limit: Optional[int],
    indices: Optional[Sequence[int]],
    resume: bool,
    aggregate_only: bool,
) -> dict[str, object]:
    """Execute or re-aggregate the shared-start optimizer comparison."""
    output_root = output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    if aggregate_only and not (output_root / "manifest.json").is_file():
        raise FileNotFoundError(
            "--aggregate-only requires an existing comparison manifest"
        )
    all_records = load_smiles(suite.input_path)
    if suite.expected_count and len(all_records) != suite.expected_count:
        raise ValueError(
            f"suite {suite.name!r} expected {suite.expected_count} records, "
            f"found {len(all_records)}"
        )
    if indices is not None:
        available_indices = {index for index, _ in all_records}
        missing_indices = sorted(set(indices) - available_indices)
        if missing_indices:
            raise ValueError(f"corpus indices do not exist: {missing_indices}")
    selected = load_smiles(suite.input_path, limit=limit, indices=indices)
    if not selected:
        raise ValueError("benchmark selection contains no molecules")
    expected_indices = tuple(index for index, _ in selected)
    scientific_configuration = {
        "backend": "hotpot",
        "workflow": WORKFLOW,
        "suite": suite.to_manifest(),
        "profile": profile.value,
        "input_sha256": sha256_file(suite.input_path),
        "settings": settings.to_manifest(),
        "selected_indices": list(expected_indices),
    }
    _write_or_check_manifest(
        output_root,
        scientific_configuration,
        workers=workers,
        resume=resume or aggregate_only,
    )

    os.environ["HOTPOT_CBOND_DEVICE"] = "cpu"
    if aggregate_only:
        previous_summary_path = output_root / "summary.json"
        previous_summary = (
            json.loads(previous_summary_path.read_text(encoding="utf-8"))
            if previous_summary_path.is_file()
            else {}
        )
        records = [
            record
            for record in load_case_reports(output_root)
            if int(record["index"]) in set(expected_indices)
        ]
        wall_seconds = previous_summary.get("wall_seconds")
    else:
        tasks = [
            (
                index,
                smiles,
                str(output_root),
                suite.metal,
                settings,
                resume,
            )
            for index, smiles in selected
        ]
        records = []
        started = perf_counter()
        context = mp.get_context("spawn")
        with ProcessPoolExecutor(max_workers=workers, mp_context=context) as pool:
            futures = {
                pool.submit(run_optimizer_comparison_case, *task): (task[0], task[1])
                for task in tasks
            }
            for completed, future in enumerate(as_completed(futures), start=1):
                index, smiles = futures[future]
                try:
                    record = future.result()
                except Exception as error:
                    record = {
                        "index": index,
                        "smiles": smiles,
                        "workflow": WORKFLOW,
                        "status": "failed_worker",
                        "phase": "worker",
                        "error_type": type(error).__name__,
                        "error_message": str(error),
                        "traceback": traceback.format_exc(),
                        "total_seconds": 0.0,
                    }
                    case_dir = output_root / "cases" / f"{index:04d}"
                    case_dir.mkdir(parents=True, exist_ok=True)
                    write_json(case_dir / "report.json", record)
                records.append(record)
                print(
                    f"[{completed}/{len(tasks)}] case={index:04d} "
                    f"status={record['status']} "
                    f"seconds={float(record['total_seconds']):.1f}",
                    flush=True,
                )
        wall_seconds = perf_counter() - started

    return aggregate_optimizer_comparison(
        output_root,
        records,
        suite,
        settings,
        profile,
        expected_indices,
        wall_seconds=wall_seconds,
    )


def _indices(value: str) -> tuple[int, ...]:
    return tuple(int(item.strip()) for item in value.split(",") if item.strip())


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Compare Hotpot complex-aware and ordinary force-field optimizers "
            "from one verified shared complex starting structure."
        )
    )
    parser.add_argument(
        "--suite", choices=tuple(BUILTIN_SUITES), default="extractants-eu-187"
    )
    parser.add_argument("--input", type=Path)
    parser.add_argument("--metal", default="Eu")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--profile",
        choices=tuple(profile.value for profile in RunProfile),
        default=RunProfile.STANDARD.value,
    )
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--cases", type=_indices)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--aggregate-only", action="store_true")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--steps-per-epoch", type=int)
    parser.add_argument("--timeout", type=float)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    profile = RunProfile.SMOKE if args.smoke else RunProfile(args.profile)
    limit = 1 if args.smoke and args.limit is None else args.limit
    suite = resolve_suite(args.suite, args.input, args.metal)
    settings = settings_with_overrides(
        profile,
        epochs=args.epochs,
        steps_per_epoch=args.steps_per_epoch,
        timeout=args.timeout,
    )
    summary = run_optimizer_comparison(
        suite,
        args.output,
        profile=profile,
        settings=settings,
        workers=args.workers,
        limit=limit,
        indices=args.cases,
        resume=args.resume,
        aggregate_only=args.aggregate_only,
    )
    print(
        f"completed={summary['sample_count']} "
        f"cbond_success={summary['cbond_success_count']} "
        f"shared_start_verified={summary['identical_start_verified_count']}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
