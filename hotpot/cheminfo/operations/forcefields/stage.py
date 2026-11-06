"""Shared force-field routing and controlled-pipeline stage adapter."""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass, fields, is_dataclass
from enum import Enum
from numbers import Integral, Real
from pathlib import Path
from typing import Callable, Mapping, Optional, Tuple, Union

import numpy as np

from hotpot.pipeline.contracts import (
    Artifact,
    JSONValue,
    MolecularPayload,
    MolecularRecord,
    MolecularStage,
    PipelineDefinitionError,
    PreparedMolecularStage,
    StageContext,
    StageExecutionError,
    StageResult,
    StageSpec,
    StageStatus,
)
from hotpot.pipeline.artifacts import artifact_from_file
from hotpot.pipeline._sample_control import (
    _sample_counts, _break_scope, _save_observation, _check_findings,
)

from .auto import auto_optimize
from .acceptance import _has_unreturnable_frame_failure
from .contracts import (
    Build3DReport,
    BuildAndOptimizeReport,
    ConvergenceLevel,
    ForceFieldError,
    ForceFieldSetupError,
    GeometryQualityError,
    BuildWorkerError,
    ComplexBuildError,
    ComplexBuildWorkerError,
    ForceFieldRunReport,
    ForceFieldWorkflowReport,
    TrajectoryPath,
)
from hotpot.cheminfo.structure.forcefields import TrajectoryStart
from .workflows import (
    build3d,
    build_and_optimize,
    complexes_build,
    optimize,
    optimize_complex,
)


__all__ = (
    "ForceFieldRouteOperations",
    "ForceFieldRouteOptions",
    "STAGE",
    "execute_forcefield_route",
    "get_stage",
    "json_value",
)


ForceFieldRouteReport = Union[ForceFieldRunReport, ForceFieldWorkflowReport]
_RouteOperation = Callable[..., ForceFieldRouteReport]
_BuildOperation = Callable[..., Build3DReport]


# These are existing rule names; required-value checks alone remove an output.
_UNAVAILABLE_CHECK_NAMES = frozenset((
    "coordinate_shape", "finite_coordinates", "topology_atom_count",
    "topology_atom_identity", "topology_added_atoms", "topology_original_bond",
    "topology_added_bond", "topology_added_hydrogen_degree", "forcefield_report",
    "forcefield_setup", "finite_final_energy", "finite_rms_gradient",
    "finite_max_gradient",
))
_GEOMETRY_CHECKS = (
    "coordinate_shape", "finite_coordinates", "topology", "atom_overlap",
    "atom_too_close", "bond_distance", "short_bond", "bond_length_ratio",
    "bond_ring_piercing", "bond_ring_scope_coverage",
)
_FF_CHECKS = tuple(sorted(_UNAVAILABLE_CHECK_NAMES | frozenset(_GEOMETRY_CHECKS) | {
    "backend_explosion", "forcefield_convergence", "rms_gradient", "max_gradient",
    "energy_change", "max_displacement", "stability_observations",
}))
_KNOWN_SAMPLE_FAILURE_CODES = (
    "FF-REQUEST-COORDINATES", "FF-REQUEST-METHOD", "FF-SETUP-UNAVAILABLE",
    "FF-BUILD-UNAVAILABLE", "FF-BUILD-TIMEOUT", "FF-COMPLEX-BUILD-TIMEOUT",
    "FF-NO-USABLE-FRAME", "FF-LIGAND-CANDIDATES-EXHAUSTED",
)
_FF_CODES = _KNOWN_SAMPLE_FAILURE_CODES + tuple(f"FF:{name}" for name in _FF_CHECKS)


def forcefield_error_evidence(error: ForceFieldError) -> dict[str, object]:
    evidence: dict[str, object] = {}
    if isinstance(error, ForceFieldSetupError) and error.report is not None:
        evidence["setup_report"] = error.report
    if isinstance(error, ForceFieldSetupError) and error.diagnostics is not None:
        evidence["build_diagnostics"] = error.diagnostics
    if isinstance(error, GeometryQualityError) and error.report is not None:
        evidence["quality_report"] = error.report
    if isinstance(error, (BuildWorkerError, ComplexBuildError)) and error.diagnostics is not None:
        evidence["build_diagnostics"] = error.diagnostics
    if isinstance(error, (BuildWorkerError, ComplexBuildWorkerError)):
        evidence["worker_error"] = {
            "type": error.error_type, "message": error.error_message,
            "traceback": error.worker_traceback,
        }
    if error.trajectory is not None:
        evidence["trajectory_available"] = True
    if error.ligand_build_attempts:
        evidence["ligand_build_attempt_count"] = len(error.ligand_build_attempts)
    return evidence


class _ForceFieldRequestError(ValueError):
    """Identify a bounded route precondition while preserving ValueError."""

    def __init__(self, message: str, sample_failure_code: str) -> None:
        super().__init__(message)
        self.sample_failure_code = sample_failure_code


class _StageArgumentParser(argparse.ArgumentParser):
    """Report stage syntax errors through the pipeline definition contract."""

    def error(self, message: str) -> None:
        raise PipelineDefinitionError(f"ff stage: {message}")


@dataclass(frozen=True)
class ForceFieldRouteOptions:
    """Scientific options shared by the force-field CLI and pipeline stage."""

    rebuild: bool
    optimize_only: bool
    route: str
    forcefield: Optional[str]
    algorithm: str
    epochs: int
    steps_per_epoch: int
    add_hydrogens: bool
    quality_level: str
    seed: Optional[int]
    timeout: float
    trajectory_start: Optional[TrajectoryStart]
    convergence_level: ConvergenceLevel = ConvergenceLevel.FAST


@dataclass(frozen=True)
class ForceFieldRouteOperations:
    build3d: _BuildOperation
    build_and_optimize: _RouteOperation
    auto_optimize: _RouteOperation
    complexes_build: _RouteOperation
    optimize_complex: _RouteOperation
    optimize: _RouteOperation


@dataclass(frozen=True)
class _PreparedForceFieldStage(PreparedMolecularStage):
    options: ForceFieldRouteOptions
    retain_trajectory: bool
    sample_break_on: Tuple[str, ...] = ()
    batch_break_on: Tuple[str, ...] = ()

    def execute(
        self,
        payload: MolecularPayload,
        context: StageContext,
    ) -> StageResult:
        if not payload.records:
            raise StageExecutionError("ff stage requires at least one input molecule")

        output_records = []
        reports = []
        rows = []
        batch_break = False
        for index, record in enumerate(payload.records):
            row = {"input_index": index, "output_index": None, "outcome": "not-attempted"}
            rows.append(row)
            if batch_break:
                continue
            trajectory_path = None
            if self.retain_trajectory:
                trajectory_path = context.stage_directory / "trajectory"
                if len(payload.records) > 1:
                    trajectory_path = trajectory_path / f"{index:04d}"
            try:
                report = execute_forcefield_route(
                    record.molecule, self.options, trajectory_path=trajectory_path,
                )
            except (ForceFieldError, ValueError) as error:
                code = getattr(error, "sample_failure_code", None)
                if not isinstance(error, (ForceFieldError, _ForceFieldRequestError)) or code not in _KNOWN_SAMPLE_FAILURE_CODES:
                    raise StageExecutionError(f"ff stage record {index} failed: {error}") from error
                failure = {"code": code, "type": type(error).__name__, "message": str(error)}
                if isinstance(error, ForceFieldError):
                    failure.update(json_value(forcefield_error_evidence(error)))
                    error_directory = context.stage_directory / "errors" / f"{index:04d}"
                    if error.trajectory is not None:
                        error.trajectory.write(error_directory / "trajectory")
                    for attempt_index, attempt in enumerate(error.ligand_build_attempts):
                        attempt.write(error_directory / f"ligand-{attempt_index:04d}")
                row.update(outcome="unavailable", failure=failure)
                codes = [code]
                if isinstance(error, GeometryQualityError) and error.report is not None:
                    findings = _check_findings(error.report, "FF", json_value)
                    if findings:
                        row["findings"] = findings
                        codes.extend(finding["code"] for finding in findings)
                scope = _break_scope(codes, self.sample_break_on, self.batch_break_on)
            else:
                row["report_index"] = len(reports)
                reports.append(json_value(report))
                findings = _check_findings(report.quality_report, "FF", json_value)
                if findings:
                    row["findings"] = findings
                scope = _break_scope((finding["code"] for finding in findings), self.sample_break_on, self.batch_break_on)
                unavailable = any(not check.passed and check.name in _UNAVAILABLE_CHECK_NAMES for check in report.quality_report.checks)
                if unavailable:
                    row["outcome"] = "unavailable"
                    if not _has_unreturnable_frame_failure(report.quality_report):
                        row["observation_path"] = _save_observation(index, record, context)
                elif scope == "sample":
                    row["outcome"] = "sample-break"
                    row["observation_path"] = _save_observation(index, record, context)
                else:
                    row.update(outcome="returned", output_index=len(output_records))
                    output_records.append(record)
            if scope is not None:
                row["break_scope"] = scope
            batch_break = scope == "batch"

        artifacts = ()
        for directory in ("trajectory", "errors", "observations"):
            artifacts += _artifacts_under(context.stage_directory / directory, context)
        return StageResult(
            status=StageStatus.QUALITY_FAILED if batch_break or not output_records else StageStatus.SUCCEEDED,
            payload=MolecularPayload(tuple(output_records)),
            artifacts=artifacts,
            report={"forcefield": tuple(reports), **_sample_counts(rows)},
            break_pipeline=batch_break,
        )


@dataclass(frozen=True)
class _ForceFieldStage(MolecularStage):
    def prepare(self, spec: StageSpec) -> PreparedMolecularStage:
        if spec.name != "ff":
            raise PipelineDefinitionError(
                f"force-field stage cannot prepare stage {spec.name!r}"
            )
        args = _stage_parser().parse_args(spec.argv)
        if args.trajectory_start is not None and not args.trajectory:
            raise PipelineDefinitionError(
                "ff stage: --trajectory-start requires --trajectory"
            )
        return _PreparedForceFieldStage(
            options=ForceFieldRouteOptions(
                rebuild=args.rebuild,
                optimize_only=args.optimize_only,
                route=args.route,
                forcefield=_FORCEFIELDS[args.forcefield],
                algorithm=args.algorithm,
                epochs=args.epochs,
                steps_per_epoch=args.steps_per_epoch,
                convergence_level=_CONVERGENCE_LEVELS[args.convergence_level],
                add_hydrogens=args.add_hydrogens,
                quality_level=args.quality,
                seed=args.seed,
                timeout=args.timeout,
                trajectory_start=(
                    None
                    if args.trajectory_start is None
                    else _TRAJECTORY_STARTS[args.trajectory_start]
                ),
            ),
            retain_trajectory=args.trajectory,
            sample_break_on=tuple(args.sample_break_on),
            batch_break_on=tuple(args.batch_break_on),
        )


_FORCEFIELDS = {
    "auto": None,
    "uff": "UFF",
    "mmff94": "MMFF94",
    "mmff94s": "MMFF94s",
    "gaff": "GAFF",
    "ghemical": "Ghemical",
}
_TRAJECTORY_STARTS = {
    value.value.replace("_", "-"): value for value in TrajectoryStart
}
_CONVERGENCE_LEVELS = {
    level.name.lower(): level for level in ConvergenceLevel
}


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("value must be a positive integer")
    return parsed


def _positive_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed) or parsed <= 0.0:
        raise argparse.ArgumentTypeError("value must be a finite positive number")
    return parsed


def _stage_parser() -> _StageArgumentParser:
    parser = _StageArgumentParser(add_help=False)
    geometry_mode = parser.add_mutually_exclusive_group()
    geometry_mode.add_argument("--rebuild", action="store_true")
    geometry_mode.add_argument("--optimize-only", action="store_true")
    parser.add_argument(
        "--route",
        choices=("auto", "organic", "complex"),
        default="auto",
    )
    parser.add_argument(
        "--forcefield",
        choices=tuple(_FORCEFIELDS),
        default="auto",
    )
    parser.add_argument(
        "--algorithm",
        choices=("conjugate", "steepest"),
        default="conjugate",
    )
    parser.add_argument("--epochs", type=_positive_int, default=100)
    parser.add_argument("--steps-per-epoch", type=_positive_int, default=100)
    parser.add_argument(
        "--convergence-level",
        choices=tuple(_CONVERGENCE_LEVELS),
        default="fast",
    )
    parser.add_argument(
        "--no-add-hydrogens",
        dest="add_hydrogens",
        action="store_false",
    )
    parser.set_defaults(add_hydrogens=True)
    parser.add_argument(
        "--quality",
        choices=("off", "basic", "standard", "strict"),
        default="standard",
    )
    parser.add_argument("--seed", type=int)
    parser.add_argument("--timeout", type=_positive_float, default=1000.0)
    parser.add_argument("--trajectory", action="store_true")
    parser.add_argument(
        "--trajectory-start",
        choices=tuple(_TRAJECTORY_STARTS),
    )
    parser.add_argument("--sample-break-on", action="append", choices=_FF_CODES, default=[])
    parser.add_argument("--batch-break-on", action="append", choices=_FF_CODES, default=[])
    return parser


def _default_operations() -> ForceFieldRouteOperations:
    return ForceFieldRouteOperations(
        build3d=build3d,
        build_and_optimize=build_and_optimize,
        auto_optimize=auto_optimize,
        complexes_build=complexes_build,
        optimize_complex=optimize_complex,
        optimize=optimize,
    )


def _optimization_options(
    options: ForceFieldRouteOptions,
    trajectory_path: Optional[TrajectoryPath],
) -> dict[str, object]:
    values: dict[str, object] = {
        "algorithm": options.algorithm,
        "epochs": options.epochs,
        "steps_per_epoch": options.steps_per_epoch,
        "convergence_level": options.convergence_level,
        "add_hydrogens": options.add_hydrogens,
        "quality_level": options.quality_level,
        "seed": options.seed,
        "save_movie": trajectory_path is not None,
        "trajectory_path": trajectory_path,
    }
    if options.trajectory_start is not None:
        values["trajectory_start"] = options.trajectory_start
    return values


def _build_options(
    options: ForceFieldRouteOptions,
    trajectory_path: Optional[TrajectoryPath],
) -> dict[str, object]:
    values = _optimization_options(options, trajectory_path)
    values["timeout"] = options.timeout
    return values


def _requires_build(mol: "Molecule", options: ForceFieldRouteOptions) -> bool:
    has_3d = mol.has_3d
    if options.rebuild:
        return True
    if options.optimize_only:
        if not has_3d:
            raise _ForceFieldRequestError(
                "--optimize-only requires non-coincident existing coordinates",
                "FF-REQUEST-COORDINATES",
            )
        return False
    return not has_3d


def _validate_forcefield_request(
    mol: "Molecule",
    options: ForceFieldRouteOptions,
) -> None:
    has_metal = mol.has_metal
    complex_route = options.route == "complex" or (
        options.route == "auto" and has_metal
    )
    if complex_route and options.forcefield not in {None, "UFF"}:
        raise _ForceFieldRequestError(
            "Complex force-field workflows currently support only UFF; "
            "use --forcefield auto or --forcefield uff",
            "FF-REQUEST-METHOD",
        )


def execute_forcefield_route(
    mol: "Molecule",
    options: ForceFieldRouteOptions,
    *,
    trajectory_path: Optional[TrajectoryPath] = None,
    operations: Optional[ForceFieldRouteOperations] = None,
) -> ForceFieldRouteReport:
    """Execute exactly one public force-field route operation on ``mol``."""
    selected = operations or _default_operations()
    _validate_forcefield_request(mol, options)
    requires_build = _requires_build(mol, options)
    has_metal = mol.has_metal

    if options.route == "auto":
        if options.optimize_only:
            optimizer = selected.optimize_complex if has_metal else selected.optimize
            return optimizer(
                mol,
                options.forcefield,
                **_optimization_options(options, trajectory_path),
            )
        if has_metal:
            return selected.auto_optimize(
                mol,
                options.forcefield,
                **_build_options(options, trajectory_path),
            )
        if requires_build:
            return selected.build_and_optimize(
                mol,
                options.forcefield,
                **_build_options(options, trajectory_path),
            )
        return selected.auto_optimize(
            mol,
            options.forcefield,
            **_optimization_options(options, trajectory_path),
        )

    if options.route == "complex":
        if requires_build:
            return selected.complexes_build(
                mol,
                options.forcefield,
                **_build_options(options, trajectory_path),
            )
        return selected.optimize_complex(
            mol,
            options.forcefield,
            **_optimization_options(options, trajectory_path),
        )

    if requires_build:
        build_report = selected.build3d(
            mol,
            add_hydrogens=options.add_hydrogens,
            seed=options.seed,
            timeout=options.timeout,
        )
        optimization_report = selected.optimize(
            mol,
            options.forcefield,
            **{
                **_optimization_options(options, trajectory_path),
                "add_hydrogens": False,
            },
        )
        quality_report = optimization_report.quality_report
        if quality_report is None:
            raise RuntimeError("The force-field optimizer omitted its quality report")
        return BuildAndOptimizeReport(
            effective_forcefield=optimization_report.effective_forcefield,
            build=build_report,
            optimization=optimization_report,
            quality_report=quality_report,
            trajectory=optimization_report.trajectory,
        )
    return selected.optimize(
        mol,
        options.forcefield,
        **_optimization_options(options, trajectory_path),
    )


def json_value(value: object) -> JSONValue:
    """Convert a typed force-field report value to strict JSON data."""
    if value is None or isinstance(value, (bool, str)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Integral):
        return int(value)
    if isinstance(value, Real):
        number = float(value)
        return number if math.isfinite(number) else None
    if isinstance(value, Enum):
        return json_value(value.value)
    if isinstance(value, np.ndarray):
        return tuple(json_value(item) for item in value.tolist())
    if is_dataclass(value) and not isinstance(value, type):
        return {
            field.name: json_value(getattr(value, field.name))
            for field in fields(value)
            if field.name != "trajectory"
            and not (
                field.name == "quality_report"
                and isinstance(value, ForceFieldWorkflowReport)
                and value.optimization is not None
            )
        }
    if isinstance(value, Mapping):
        return {str(key): json_value(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return tuple(json_value(item) for item in value)
    raise TypeError(f"Cannot serialize force-field value {type(value)!r}")


def _artifacts_under(path: Path, context: StageContext) -> Tuple[Artifact, ...]:
    if not path.exists():
        return ()
    return tuple(
        artifact_from_file(context.stage_directory, item)
        for item in sorted(path.rglob("*"))
        if item.is_file()
    )


STAGE = _ForceFieldStage()


def get_stage() -> MolecularStage:
    """Return the registered force-field molecular stage."""
    return STAGE
