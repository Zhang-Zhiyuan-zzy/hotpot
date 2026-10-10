"""Shared force-field routing and controlled-pipeline stage adapter."""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass, fields, is_dataclass
from enum import Enum
from numbers import Integral, Real
from pathlib import Path
from typing import TYPE_CHECKING, Callable, Mapping, Optional, Tuple, Union

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

from .auto import auto_optimize
from .contracts import (
    Build3DReport,
    BuildAndOptimizeReport,
    ConvergenceLevel,
    ForceFieldError,
    ForceFieldRunReport,
    ForceFieldWorkflowReport,
    TrajectoryPath,
)
from .trajectory import TrajectoryStart
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


if TYPE_CHECKING:
    from hotpot.cheminfo.core import Molecule


ForceFieldRouteReport = Union[ForceFieldRunReport, ForceFieldWorkflowReport]
_RouteOperation = Callable[..., ForceFieldRouteReport]
_BuildOperation = Callable[..., Build3DReport]


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

    def execute(
        self,
        payload: MolecularPayload,
        context: StageContext,
    ) -> StageResult:
        if not payload.records:
            raise StageExecutionError("ff stage requires at least one input molecule")

        output_records = []
        reports = []
        quality_failed = False
        for index, record in enumerate(payload.records):
            trajectory_path = None
            if self.retain_trajectory:
                trajectory_path = context.stage_directory / "trajectory"
                if len(payload.records) > 1:
                    trajectory_path = trajectory_path / f"{index:04d}"
            try:
                report = execute_forcefield_route(
                    record.molecule,
                    self.options,
                    trajectory_path=trajectory_path,
                )
            except (ForceFieldError, ValueError) as error:
                raise StageExecutionError(
                    f"ff stage record {index} failed: {error}"
                ) from error
            quality_failed = quality_failed or not report.quality_report.passed
            reports.append(json_value(report))
            output_records.append(
                MolecularRecord(record.molecule, record.electronic_state)
            )

        artifacts = _artifacts_under(context.stage_directory / "trajectory", context)
        status = (
            StageStatus.QUALITY_FAILED
            if quality_failed
            else StageStatus.SUCCEEDED
        )
        stderr = (
            "One or more force-field structures failed the requested quality profile."
            if quality_failed
            else ""
        )
        return StageResult(
            status=status,
            payload=MolecularPayload(tuple(output_records)),
            artifacts=artifacts,
            report={"forcefield": tuple(reports)},
            stderr=stderr,
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
            raise ValueError(
                "--optimize-only requires non-coincident existing coordinates"
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
        raise ValueError(
            "Complex force-field workflows currently support only UFF; "
            "use --forcefield auto or --forcefield uff"
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
            requested_forcefield=optimization_report.requested_forcefield,
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
