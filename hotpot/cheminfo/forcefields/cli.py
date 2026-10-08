"""Command-line adapter for Hotpot force-field workflows."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, fields, is_dataclass
from enum import Enum
from importlib import resources
from multiprocessing import get_context
from numbers import Integral, Real
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Mapping, Optional, Sequence, TYPE_CHECKING, Tuple, Union

from hotpot._cli import MarkdownDocumentationAction
from hotpot.cheminfo._io import MolReader

from . import (
    BuildAndOptimizeReport,
    BuildWorkerError,
    ComplexBuildError,
    ComplexBuildWorkerError,
    ConvergenceLevel,
    ForceFieldError,
    ForceFieldSetupError,
    GeometryQualityError,
    TrajectoryStart,
    auto_optimize,
    build3d,
    build_and_optimize,
    complexes_build,
    optimize,
    optimize_complex,
)


if TYPE_CHECKING:
    from ..core import Molecule


__all__ = (
    "add_arguments",
    "build_parser",
    "load_cli_documentation",
    "main",
    "run",
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


class _CLIUsageError(ValueError):
    """A command request that is invalid before force-field execution."""


@dataclass(frozen=True)
class _InputRecord:
    index: int
    source: str
    source_record_index: int
    mol: "Molecule"


@dataclass(frozen=True)
class _RunOptions:
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
    output_format: str
    convergence_level: ConvergenceLevel = ConvergenceLevel.FAST


@dataclass(frozen=True)
class _WorkItem:
    record: _InputRecord
    options: _RunOptions
    trajectory_path: Optional[str]


@dataclass(frozen=True)
class _Outcome:
    index: int
    source: str
    source_record_index: int
    status: str
    payload: Optional[str]
    report: Optional[object]
    error_type: Optional[str] = None
    error_message: Optional[str] = None
    error_evidence: Optional[Mapping[str, object]] = None


def load_cli_documentation() -> str:
    """Return the packaged long-form command documentation."""
    return (
        resources.files(__package__)
        .joinpath("cli_doc.md")
        .read_text(encoding="utf-8")
    )


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


def _load_molecules(
    source: Union[str, Path],
    input_format: Optional[str],
    source_name: str,
) -> tuple["Molecule", ...]:
    try:
        return tuple(MolReader(source, fmt=input_format))
    except OSError as error:
        raise _CLIUsageError(
            f"Could not read molecule input {source_name!r}: {error}"
        ) from error


def add_arguments(parser: argparse.ArgumentParser) -> None:
    """Register the ``hotpot ff`` public command arguments."""
    parser.add_argument(
        "--doc",
        action=MarkdownDocumentationAction,
        nargs=0,
        document_loader=load_cli_documentation,
        help="show detailed Markdown usage documentation and exit",
    )
    parser.add_argument(
        "inputs",
        nargs="+",
        metavar="SMILES/FILE",
        help="one or more SMILES strings, molecule files, or '-' for stdin",
    )
    parser.add_argument(
        "-o",
        "--output",
        help="write the molecular payload to FILE instead of stdout; '-' means stdout",
    )
    parser.add_argument(
        "--input-format",
        help="input format override, such as smi, sdf, or mol2",
    )
    parser.add_argument(
        "--output-format",
        help="output molecule format (default: output suffix, otherwise mol2)",
    )
    geometry_mode = parser.add_mutually_exclusive_group()
    geometry_mode.add_argument(
        "--rebuild",
        action="store_true",
        help="discard existing coordinates and build a new 3D starting geometry",
    )
    geometry_mode.add_argument(
        "--optimize-only",
        action="store_true",
        help="optimize existing 3D coordinates and fail when they are absent",
    )
    parser.add_argument(
        "--route",
        choices=("auto", "organic", "complex"),
        default="auto",
        help="force-field workflow route (default: auto from molecular topology)",
    )
    parser.add_argument(
        "--forcefield",
        choices=tuple(_FORCEFIELDS),
        default="auto",
        help="force-field backend (default: auto; complexes currently require UFF)",
    )
    parser.add_argument(
        "--algorithm",
        choices=("conjugate", "steepest"),
        default="conjugate",
        help="Open Babel optimization algorithm (default: conjugate)",
    )
    parser.add_argument(
        "--epochs",
        type=_positive_int,
        default=100,
        help="optimization epochs (default: 100)",
    )
    parser.add_argument(
        "--steps-per-epoch",
        type=_positive_int,
        default=100,
        help="Open Babel steps submitted per epoch (default: 100)",
    )
    parser.add_argument(
        "--convergence-level",
        choices=tuple(_CONVERGENCE_LEVELS),
        default="fast",
        help="optimizer convergence evidence level (default: fast)",
    )
    parser.add_argument(
        "--no-add-hydrogens",
        dest="add_hydrogens",
        action="store_false",
        help="do not add missing hydrogens before building or optimization",
    )
    parser.set_defaults(add_hydrogens=True)
    parser.add_argument(
        "--quality",
        choices=("off", "basic", "standard", "strict"),
        default="standard",
        help="terminal structure quality gate (default: standard)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        help="seed Open Babel building and Hotpot perturbations",
    )
    parser.add_argument(
        "--timeout",
        type=_positive_float,
        default=1000.0,
        help="coordinate-build timeout in seconds (default: 1000)",
    )
    parser.add_argument(
        "--trajectory",
        metavar="DIRECTORY",
        help="persist the complete topology-aware force-field trajectory",
    )
    parser.add_argument(
        "--trajectory-start",
        choices=tuple(_TRAJECTORY_STARTS),
        help="earliest trajectory stage to retain (default: workflow-specific)",
    )
    parser.add_argument(
        "--report",
        metavar="FILE",
        help="write a JSON execution and scientific-quality report",
    )
    parser.add_argument(
        "--jobs",
        type=_positive_int,
        default=1,
        help="independent molecule worker processes (default: 1)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="replace existing output, report, and trajectory paths",
    )


def _read_molecules(
    sources: Sequence[str],
    input_format: Optional[str],
) -> Tuple[_InputRecord, ...]:
    if sources.count("-") > 1:
        raise _CLIUsageError("stdin '-' may be specified only once")

    records = []
    for source in sources:
        if source == "-":
            content = sys.stdin.read()
            if not content.strip():
                raise _CLIUsageError("stdin did not contain a molecule")
            stdin_format = input_format or "smi"
            with TemporaryDirectory(prefix="hotpot-ff-stdin-") as directory:
                stdin_path = Path(directory) / f"stdin.{stdin_format}"
                stdin_path.write_text(content, encoding="utf-8")
                molecules = _load_molecules(stdin_path, stdin_format, "stdin")
            source_name = "stdin"
        elif os.path.isfile(source):
            source_name = source
            molecules = _load_molecules(source, input_format, source_name)
        else:
            source_name = source
            molecules = _load_molecules(
                source,
                input_format or "smi",
                source_name,
            )

        if not molecules:
            raise _CLIUsageError(f"No molecules were found in {source_name!r}")
        for source_record_index, mol in enumerate(molecules):
            records.append(
                _InputRecord(
                    index=len(records),
                    source=source_name,
                    source_record_index=source_record_index,
                    mol=mol,
                )
            )
    return tuple(records)


def _infer_output_format(
    output: Optional[str],
    output_format: Optional[str],
) -> str:
    if output_format:
        return output_format.lower().lstrip(".")
    if output and output != "-":
        suffix = Path(output).suffix
        if suffix:
            return suffix[1:].lower()
    return "mol2"


def _trajectory_path(
    base_path: Optional[str],
    record: _InputRecord,
    record_count: int,
) -> Optional[str]:
    if base_path is None:
        return None
    if record_count == 1:
        return base_path
    return str(Path(base_path) / f"{record.index:04d}")


def _optimization_options(
    options: _RunOptions,
    trajectory_path: Optional[str],
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
    options: _RunOptions,
    trajectory_path: Optional[str],
) -> dict[str, object]:
    values = _optimization_options(options, trajectory_path)
    values["timeout"] = options.timeout
    return values


def _requires_build(mol: "Molecule", options: _RunOptions) -> bool:
    if options.rebuild:
        return True
    if options.optimize_only:
        if not mol.has_3d:
            raise _CLIUsageError(
                "--optimize-only requires non-coincident existing coordinates"
            )
        return False
    return not mol.has_3d


def _validate_forcefield_request(mol: "Molecule", options: _RunOptions) -> None:
    complex_route = options.route == "complex" or (
        options.route == "auto" and mol.has_metal
    )
    if complex_route and options.forcefield not in {None, "UFF"}:
        raise _CLIUsageError(
            "Complex force-field workflows currently support only UFF; "
            "use --forcefield auto or --forcefield uff"
        )


def _run_forcefield(
    mol: "Molecule",
    options: _RunOptions,
    trajectory_path: Optional[str],
) -> object:
    _validate_forcefield_request(mol, options)
    requires_build = _requires_build(mol, options)
    if options.route == "auto":
        if requires_build:
            return build_and_optimize(
                mol,
                options.forcefield,
                **_build_options(options, trajectory_path),
            )
        return auto_optimize(
            mol,
            options.forcefield,
            **_optimization_options(options, trajectory_path),
        )

    if options.route == "complex":
        if requires_build:
            return complexes_build(
                mol,
                options.forcefield,
                **_build_options(options, trajectory_path),
            )
        return optimize_complex(
            mol,
            options.forcefield,
            **_optimization_options(options, trajectory_path),
        )

    if requires_build:
        build_report = build3d(
            mol,
            add_hydrogens=options.add_hydrogens,
            seed=options.seed,
            timeout=options.timeout,
        )
        optimization_report = optimize(
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
    return optimize(
        mol,
        options.forcefield,
        **_optimization_options(options, trajectory_path),
    )


def _normalize_molecule_payload(text: str) -> str:
    return text.rstrip("\n") + "\n"


def _forcefield_error_evidence(error: ForceFieldError) -> dict[str, object]:
    evidence: dict[str, object] = {}
    if isinstance(error, ForceFieldSetupError) and error.report is not None:
        evidence["setup_report"] = error.report
    if isinstance(error, ForceFieldSetupError) and error.diagnostics is not None:
        evidence["build_diagnostics"] = error.diagnostics
    if isinstance(error, GeometryQualityError) and error.report is not None:
        evidence["quality_report"] = error.report
    if (
        isinstance(error, (BuildWorkerError, ComplexBuildError))
        and error.diagnostics is not None
    ):
        evidence["build_diagnostics"] = error.diagnostics
    if isinstance(error, (BuildWorkerError, ComplexBuildWorkerError)):
        evidence["worker_error"] = {
            "type": error.error_type,
            "message": error.error_message,
            "traceback": error.worker_traceback,
        }
    if error.trajectory is not None:
        evidence["trajectory_available"] = True
    if error.ligand_build_attempts:
        evidence["ligand_build_attempt_count"] = len(error.ligand_build_attempts)
    return evidence


def _process_work_item(item: _WorkItem) -> _Outcome:
    record = item.record
    try:
        report = _run_forcefield(record.mol, item.options, item.trajectory_path)
    except _CLIUsageError as error:
        return _Outcome(
            index=record.index,
            source=record.source,
            source_record_index=record.source_record_index,
            status="error",
            payload=None,
            report=None,
            error_type="InputError",
            error_message=str(error),
        )
    except ForceFieldError as error:
        return _Outcome(
            index=record.index,
            source=record.source,
            source_record_index=record.source_record_index,
            status="error",
            payload=None,
            report=None,
            error_type=type(error).__name__,
            error_message=str(error),
            error_evidence=_forcefield_error_evidence(error),
        )

    quality_report = report.quality_report
    status = "ok" if quality_report.passed else "quality-failed"
    payload = _normalize_molecule_payload(
        record.mol.write(fmt=item.options.output_format, write_single=True)
    )
    return _Outcome(
        index=record.index,
        source=record.source,
        source_record_index=record.source_record_index,
        status=status,
        payload=payload,
        report=report,
    )


def _process_work_items(
    items: Sequence[_WorkItem],
    jobs: int,
) -> Tuple[_Outcome, ...]:
    if jobs == 1:
        return tuple(_process_work_item(item) for item in items)
    with ProcessPoolExecutor(
        max_workers=jobs,
        mp_context=get_context("spawn"),
    ) as executor:
        return tuple(executor.map(_process_work_item, items, chunksize=1))


def _json_value(value: object) -> object:
    if value is None or isinstance(value, (bool, str)):
        return value
    if isinstance(value, Integral):
        return int(value)
    if isinstance(value, Real):
        number = float(value)
        return number if math.isfinite(number) else None
    if isinstance(value, Enum):
        return _json_value(value.value)
    if is_dataclass(value) and not isinstance(value, type):
        return {
            field.name: _json_value(getattr(value, field.name))
            for field in fields(value)
            if field.name != "trajectory"
        }
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_value(item) for item in value]
    raise TypeError(f"Cannot serialize force-field report value {type(value)!r}")


def _report_payload(outcomes: Sequence[_Outcome]) -> str:
    results = []
    for outcome in outcomes:
        result: dict[str, object] = {
            "index": outcome.index,
            "source": outcome.source,
            "source_record_index": outcome.source_record_index,
            "status": outcome.status,
        }
        if outcome.report is not None:
            result["forcefield_report"] = _json_value(outcome.report)
        if outcome.error_type is not None:
            error: dict[str, object] = {
                "type": outcome.error_type,
                "message": outcome.error_message,
            }
            if outcome.error_evidence:
                error["evidence"] = _json_value(outcome.error_evidence)
            result["error"] = error
        results.append(result)
    return json.dumps(
        {"schema_version": 1, "results": results},
        indent=2,
        sort_keys=True,
        allow_nan=False,
    ) + "\n"


def _check_output_paths(args: argparse.Namespace) -> None:
    named_paths = {
        name: Path(path)
        for name, path in (
            ("--output", args.output),
            ("--report", args.report),
            ("--trajectory", args.trajectory),
        )
        if path and path != "-"
    }
    paths = tuple(named_paths.values())
    resolved_paths = tuple(
        (name, path.expanduser().resolve())
        for name, path in named_paths.items()
    )
    for index, (left_name, left_path) in enumerate(resolved_paths):
        for right_name, right_path in resolved_paths[index + 1 :]:
            if (
                left_path == right_path
                or left_path in right_path.parents
                or right_path in left_path.parents
            ):
                raise _CLIUsageError(
                    f"{left_name} and {right_name} paths must not overlap"
                )
    if args.report == "-":
        raise _CLIUsageError(
            "--report - would collide with the molecular stdout payload"
        )
    if args.trajectory == "-":
        raise _CLIUsageError("--trajectory requires a directory path, not '-'")
    if args.trajectory_start is not None and args.trajectory is None:
        raise _CLIUsageError("--trajectory-start requires --trajectory")
    for option_name in ("output", "report"):
        path = getattr(args, option_name)
        if path and path != "-" and Path(path).is_dir():
            raise _CLIUsageError(f"--{option_name} requires a file path")
    if args.trajectory and Path(args.trajectory).is_file():
        raise _CLIUsageError("--trajectory requires a directory path")
    if not args.overwrite:
        existing = tuple(path for path in paths if path.exists())
        if existing:
            joined = ", ".join(str(path) for path in existing)
            raise _CLIUsageError(
                f"Refusing to replace existing path(s): {joined}; use --overwrite"
            )


def _write_text(path: str, text: str) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(text, encoding="utf-8")


def run(args: argparse.Namespace) -> int:
    """Execute one parsed ``hotpot ff`` command."""
    try:
        _check_output_paths(args)
        records = _read_molecules(args.inputs, args.input_format)
    except _CLIUsageError as error:
        print(f"hotpot ff: {error}", file=sys.stderr)
        return 2
    output_format = _infer_output_format(args.output, args.output_format)
    trajectory_start = (
        None
        if args.trajectory_start is None
        else _TRAJECTORY_STARTS[args.trajectory_start]
    )
    options = _RunOptions(
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
        trajectory_start=trajectory_start,
        output_format=output_format,
    )
    items = tuple(
        _WorkItem(
            record=record,
            options=options,
            trajectory_path=_trajectory_path(args.trajectory, record, len(records)),
        )
        for record in records
    )
    outcomes = _process_work_items(items, args.jobs)
    molecular_payload = "".join(
        outcome.payload for outcome in outcomes if outcome.payload is not None
    )

    if args.output and args.output != "-":
        if molecular_payload:
            _write_text(args.output, molecular_payload)
        elif args.overwrite:
            Path(args.output).unlink(missing_ok=True)
    else:
        sys.stdout.write(molecular_payload)

    if args.report:
        _write_text(args.report, _report_payload(outcomes))

    for outcome in outcomes:
        if outcome.error_type is not None:
            print(
                f"hotpot ff: {outcome.source} record "
                f"{outcome.source_record_index}: {outcome.error_type}: "
                f"{outcome.error_message}",
                file=sys.stderr,
            )
        elif outcome.status == "quality-failed":
            print(
                f"hotpot ff: {outcome.source} record "
                f"{outcome.source_record_index}: quality profile "
                f"{args.quality!r} failed; the inspectable structure was emitted"
                + (f"; see {args.report}" if args.report else ""),
                file=sys.stderr,
            )
    return 0 if all(outcome.status == "ok" for outcome in outcomes) else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build and optimize molecular geometries with force fields"
    )
    add_arguments(parser)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    return run(build_parser().parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
