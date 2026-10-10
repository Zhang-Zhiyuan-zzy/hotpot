"""Command-line adapter for independent GFN-FF and GFN-xTB nodes."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, fields, is_dataclass
from enum import Enum
from importlib import resources
from numbers import Integral, Real
from pathlib import Path
from typing import Mapping, Optional, Sequence, Tuple

from hotpot._cli import MarkdownDocumentationAction
from hotpot.cheminfo._io import MolReader
from hotpot.cheminfo.calculator.electronic_state import (
    ChargeInferenceResult,
    ElectronicState,
    ElectronicStateError,
)
from hotpot.cheminfo.calculator.electronic_state.resolver import (
    resolve_electronic_state,
)
from hotpot.cheminfo.calculator.formal_charges import infer_charge
from hotpot.cheminfo.core import Molecule

from .contracts import (
    GFNXTBMethod,
    XTBError,
    XTBExecutionError,
    XTBMethod,
    XTBResultError,
    XTBRunReport,
    XTBTask,
)
from .stream import (
    XTBStreamError,
    XTBStreamMetadata,
    XTBStreamRecord,
    metadata_from_report,
    read_sdf_records,
    write_sdf_records,
)
from .workflow import run_gfn_xtb, run_gfnff


__all__ = (
    "add_arguments",
    "build_parser",
    "load_cli_documentation",
    "main",
    "run",
)


_METHODS = {
    "gfnff": XTBMethod.GFNFF,
    "gfn0": XTBMethod.GFN0_XTB,
    "gfn1": XTBMethod.GFN1_XTB,
    "gfn2": XTBMethod.GFN2_XTB,
}
_GFN_XTB_METHODS = {
    "gfn0": GFNXTBMethod.GFN0_XTB,
    "gfn1": GFNXTBMethod.GFN1_XTB,
    "gfn2": GFNXTBMethod.GFN2_XTB,
}
_TASKS = {
    "singlepoint": XTBTask.SINGLEPOINT,
    "optimize": XTBTask.OPTIMIZE,
}
_CHARGE_MODELS = ("valence", "valence-constrained", "preserve")
_POST_CHECKS = ("off", "basic", "standard", "strict")


class _CLIUsageError(ValueError):
    """A command request that is invalid before xTB execution."""


@dataclass(frozen=True)
class _NamedChargeEstimator:
    model: str

    def infer(self, mol: Molecule) -> ChargeInferenceResult:
        return infer_charge(mol, model=self.model)


@dataclass(frozen=True)
class _InputRecord:
    index: int
    mol: Molecule
    metadata: Optional[XTBStreamMetadata]


@dataclass(frozen=True)
class _RunOptions:
    method: XTBMethod
    task: XTBTask
    charge: Optional[int]
    unpaired_electrons: Optional[int]
    charge_model: str
    output_format: str
    executable: Optional[str]
    threads: Optional[int]
    work_directory: Optional[Path]
    keep_work_directory: bool
    post_check: str


@dataclass(frozen=True)
class _WorkItem:
    record: _InputRecord
    options: _RunOptions


@dataclass(frozen=True)
class _Outcome:
    index: int
    status: str
    payload: Optional[str]
    report: Optional[XTBRunReport]
    quality_report: Optional[object]
    native_stdout: str
    native_stderr: str
    error_type: Optional[str] = None
    error_message: Optional[str] = None


def load_cli_documentation() -> str:
    """Return the packaged long-form xTB command documentation."""
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


def _nonnegative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("value must be a nonnegative integer")
    return parsed


def add_arguments(parser: argparse.ArgumentParser) -> None:
    """Register the ``hotpot xtb`` public command arguments."""
    parser.add_argument(
        "--doc",
        action=MarkdownDocumentationAction,
        nargs=0,
        document_loader=load_cli_documentation,
        help="show detailed Markdown usage documentation and exit",
    )
    parser.add_argument(
        "input",
        nargs="?",
        metavar="3D-STRUCTURE-FILE/-",
        help="3D molecular file or '-' for standard input",
    )
    parser.add_argument(
        "--method",
        choices=tuple(_METHODS),
        default="gfn2",
        help="numerical method (default: gfn2)",
    )
    parser.add_argument(
        "--task",
        choices=tuple(_TASKS),
        default="optimize",
        help="calculation task (default: optimize)",
    )
    parser.add_argument("--charge", type=int, help="authoritative total charge")
    parser.add_argument(
        "--unpaired-electrons",
        type=_nonnegative_int,
        help="authoritative xTB UHF/unpaired-electron count",
    )
    parser.add_argument(
        "--charge-model",
        choices=_CHARGE_MODELS,
        default="valence",
        help="default formal-charge inference model (default: valence)",
    )
    parser.add_argument(
        "--input-format",
        help="input format override; required for standard input",
    )
    parser.add_argument(
        "--output-format",
        default="sdf",
        help="molecular output format (default: sdf)",
    )
    parser.add_argument(
        "-o",
        "--output",
        help="write molecular payload to FILE; '-' means standard output",
    )
    parser.add_argument("--report", help="write a structured JSON report to FILE")
    parser.add_argument(
        "--native-log",
        help="write captured native xTB stdout and stderr to FILE",
    )
    parser.add_argument(
        "--xtb-executable",
        help="path to the official xTB executable",
    )
    parser.add_argument(
        "--threads",
        type=_positive_int,
        help="threads assigned to each xTB process",
    )
    parser.add_argument(
        "--jobs",
        type=_positive_int,
        default=1,
        help="independent xTB molecule processes (default: 1)",
    )
    parser.add_argument(
        "--work-directory",
        help="parent directory for isolated native workspaces",
    )
    parser.add_argument(
        "--keep-work-directory",
        action="store_true",
        help="retain native workspaces and artifacts",
    )
    parser.add_argument(
        "--post-check",
        choices=_POST_CHECKS,
        default="off",
        help="optional Hotpot geometry gate after xTB (default: off)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="replace existing output, report, and native-log files",
    )


def _normalized_format(value: str) -> str:
    return value.lower().lstrip(".")


def _input_format(source: str, requested: Optional[str]) -> str:
    if requested:
        return _normalized_format(requested)
    if source == "-":
        raise _CLIUsageError("--input-format is required for standard input")
    suffix = Path(source).suffix
    if not suffix:
        raise _CLIUsageError("input format cannot be inferred from the file name")
    return _normalized_format(suffix)


def _read_records(
    source: Optional[str],
    requested_format: Optional[str],
) -> Tuple[_InputRecord, ...]:
    if source is None:
        raise _CLIUsageError("a 3D structure file or '-' is required")
    input_format = _input_format(source, requested_format)
    if source == "-":
        content = sys.stdin.read()
        if not content.strip():
            raise _CLIUsageError("standard input did not contain a molecule")
    else:
        input_path = Path(source)
        if not input_path.is_file():
            raise _CLIUsageError(f"molecule input file does not exist: {source}")
        content = input_path.read_text(encoding="utf-8")

    if input_format in {"smi", "smiles", "can"}:
        raise _CLIUsageError(
            "SMILES does not contain the complete explicit-atom 3D coordinates "
            "required by the independent xTB node"
        )

    if input_format == "sdf":
        try:
            stream_records = read_sdf_records(content)
        except XTBStreamError as error:
            raise _CLIUsageError(str(error)) from error
        return tuple(
            _InputRecord(index, record.mol, record.metadata)
            for index, record in enumerate(stream_records)
        )

    try:
        molecules = tuple(MolReader(content, fmt=input_format))
    except (OSError, RuntimeError, ValueError) as error:
        raise _CLIUsageError(f"could not read molecular input: {error}") from error
    if not molecules:
        raise _CLIUsageError("molecular input did not contain any records")
    return tuple(
        _InputRecord(index, mol, None)
        for index, mol in enumerate(molecules)
    )


def _check_output_paths(args: argparse.Namespace) -> None:
    if args.report == "-":
        raise _CLIUsageError("--report cannot share the molecular stdout stream")
    if args.native_log == "-":
        raise _CLIUsageError("--native-log cannot share the molecular stdout stream")

    named_paths = {
        name: Path(value).expanduser().resolve()
        for name, value in (
            ("--output", args.output),
            ("--report", args.report),
            ("--native-log", args.native_log),
        )
        if value and value != "-"
    }
    resolved = tuple(named_paths.items())
    for index, (left_name, left_path) in enumerate(resolved):
        for right_name, right_path in resolved[index + 1 :]:
            if left_path == right_path:
                raise _CLIUsageError(
                    f"{left_name} and {right_name} must use different files"
                )
    for option, path in resolved:
        if path.is_dir():
            raise _CLIUsageError(f"{option} requires a file path")
        if path.exists() and not args.overwrite:
            raise _CLIUsageError(
                f"refusing to replace {path}; use --overwrite"
            )


def _environment(threads: Optional[int]) -> Mapping[str, str]:
    environment = dict(os.environ)
    if threads is not None:
        thread_count = str(threads)
        environment.update(
            {
                "OMP_NUM_THREADS": thread_count,
                "MKL_NUM_THREADS": thread_count,
                "OPENBLAS_NUM_THREADS": thread_count,
            }
        )
    return environment


def _state_values(
    record: _InputRecord,
    options: _RunOptions,
) -> tuple[Optional[int], Optional[int]]:
    metadata = record.metadata
    charge = options.charge
    unpaired = options.unpaired_electrons
    if charge is None and metadata is not None:
        charge = metadata.total_charge
    if unpaired is None and metadata is not None:
        unpaired = metadata.unpaired_electrons
    return charge, unpaired


def _electronic_state(
    record: _InputRecord,
    options: _RunOptions,
    charge: Optional[int],
    unpaired_electrons: Optional[int],
) -> ElectronicState:
    metadata = record.metadata
    if (
        metadata is not None
        and options.charge is None
        and options.unpaired_electrons is None
        and metadata.unpaired_electrons is not None
    ):
        return ElectronicState(
            charge=metadata.total_charge,
            unpaired_electrons=metadata.unpaired_electrons,
            multiplicity=metadata.unpaired_electrons + 1,
            fragment_charges=(),
            charge_source=metadata.provenance.charge_source,
            spin_source=metadata.provenance.spin_source,
            assumptions=("Reused validated electronic state from the input SDF record.",),
        )
    estimator = (
        None if charge is not None else _NamedChargeEstimator(options.charge_model)
    )
    return resolve_electronic_state(
        record.mol,
        charge=charge,
        unpaired_electrons=unpaired_electrons,
        charge_estimator=estimator,
    )


def _run_xtb_record(
    record: _InputRecord,
    options: _RunOptions,
) -> XTBRunReport:
    charge, unpaired_electrons = _state_values(record, options)
    common = {
        "task": options.task,
        "executable": options.executable,
        "environment": _environment(options.threads),
        "work_directory": options.work_directory,
        "keep_work_directory": options.keep_work_directory,
    }
    if options.method is XTBMethod.GFNFF:
        if options.unpaired_electrons is not None:
            raise _CLIUsageError(
                "GFN-FF does not accept --unpaired-electrons"
            )
        if charge is not None:
            return run_gfnff(record.mol, charge=charge, **common)
        return run_gfnff(
            record.mol,
            charge_state=infer_charge(record.mol, model=options.charge_model),
            **common,
        )

    state = _electronic_state(
        record,
        options,
        charge,
        unpaired_electrons,
    )
    return run_gfn_xtb(
        record.mol,
        method=_GFN_XTB_METHODS[options.method.value],
        state=state,
        **common,
    )


def _post_check(mol: Molecule, level: str) -> Optional[object]:
    if level == "off":
        return None
    from hotpot.cheminfo.forcefields import evaluate_structure_acceptance

    return evaluate_structure_acceptance(mol, level=level)


def _serialize_record(
    record: _InputRecord,
    report: XTBRunReport,
    output_format: str,
) -> str:
    if output_format == "sdf":
        metadata = metadata_from_report(report, record.metadata)
        return write_sdf_records((XTBStreamRecord(record.mol, metadata),))
    return record.mol.write(fmt=output_format, write_single=True).rstrip("\n") + "\n"


def _error_report(error: XTBError) -> Optional[XTBRunReport]:
    if isinstance(error, (XTBExecutionError, XTBResultError)):
        return error.report
    return None


def _process_work_item(item: _WorkItem) -> _Outcome:
    record = item.record
    try:
        report = _run_xtb_record(record, item.options)
        quality_report = _post_check(record.mol, item.options.post_check)
        status = (
            "quality-failed"
            if quality_report is not None and not quality_report.passed
            else "ok"
        )
        payload = _serialize_record(
            record,
            report,
            item.options.output_format,
        )
        return _Outcome(
            index=record.index,
            status=status,
            payload=payload,
            report=report,
            quality_report=quality_report,
            native_stdout=report.stdout,
            native_stderr=report.stderr,
        )
    except (XTBError, XTBStreamError, ElectronicStateError, _CLIUsageError, ValueError) as error:
        report = _error_report(error) if isinstance(error, XTBError) else None
        native_stdout = "" if report is None else report.stdout
        native_stderr = "" if report is None else report.stderr
        message = str(error)
        if native_stderr.strip() and native_stderr.strip() not in message:
            message = f"{message}: {native_stderr.strip()}"
        return _Outcome(
            index=record.index,
            status="error",
            payload=None,
            report=report,
            quality_report=None,
            native_stdout=native_stdout,
            native_stderr=native_stderr,
            error_type=type(error).__name__,
            error_message=message,
        )


def _process_work_items(
    items: Sequence[_WorkItem],
    jobs: int,
) -> Tuple[_Outcome, ...]:
    if jobs == 1:
        return tuple(_process_work_item(item) for item in items)
    with ThreadPoolExecutor(max_workers=jobs) as executor:
        return tuple(executor.map(_process_work_item, items))


def _json_value(value: object) -> object:
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
        return _json_value(value.value)
    if is_dataclass(value) and not isinstance(value, type):
        return {
            field.name: _json_value(getattr(value, field.name))
            for field in fields(value)
        }
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_value(item) for item in value]
    raise TypeError(f"Cannot serialize xTB report value {type(value)!r}")


def _report_payload(outcomes: Sequence[_Outcome]) -> str:
    results = []
    for outcome in outcomes:
        result: dict[str, object] = {
            "index": outcome.index,
            "status": outcome.status,
        }
        if outcome.report is not None:
            result["xtb_report"] = _json_value(outcome.report)
        if outcome.quality_report is not None:
            result["quality_report"] = _json_value(outcome.quality_report)
        if outcome.error_type is not None:
            result["error"] = {
                "type": outcome.error_type,
                "message": outcome.error_message,
            }
        results.append(result)
    return json.dumps(
        {"schema_version": 1, "results": results},
        indent=2,
        sort_keys=True,
        allow_nan=False,
    ) + "\n"


def _native_log_payload(outcomes: Sequence[_Outcome]) -> str:
    sections = []
    for outcome in outcomes:
        sections.extend(
            (
                f"=== record {outcome.index} stdout ===\n{outcome.native_stdout.rstrip()}\n",
                f"=== record {outcome.index} stderr ===\n{outcome.native_stderr.rstrip()}\n",
            )
        )
    return "".join(sections)


def _write_text(path: str, text: str) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(text, encoding="utf-8")


def run(args: argparse.Namespace) -> int:
    """Execute one parsed ``hotpot xtb`` command."""
    try:
        _check_output_paths(args)
        if args.method == "gfnff" and args.unpaired_electrons is not None:
            raise _CLIUsageError("GFN-FF does not accept --unpaired-electrons")
        records = _read_records(args.input, args.input_format)
    except _CLIUsageError as error:
        print(f"hotpot xtb: {error}", file=sys.stderr)
        return 2

    options = _RunOptions(
        method=_METHODS[args.method],
        task=_TASKS[args.task],
        charge=args.charge,
        unpaired_electrons=args.unpaired_electrons,
        charge_model=args.charge_model,
        output_format=_normalized_format(args.output_format),
        executable=args.xtb_executable,
        threads=args.threads,
        work_directory=(
            None if args.work_directory is None else Path(args.work_directory)
        ),
        keep_work_directory=args.keep_work_directory,
        post_check=args.post_check,
    )
    outcomes = _process_work_items(
        tuple(_WorkItem(record, options) for record in records),
        args.jobs,
    )
    molecular_payload = "".join(
        outcome.payload for outcome in outcomes if outcome.payload is not None
    )
    if args.output and args.output != "-":
        if molecular_payload:
            _write_text(args.output, molecular_payload)
    else:
        sys.stdout.write(molecular_payload)
    if args.report:
        _write_text(args.report, _report_payload(outcomes))
    if args.native_log:
        _write_text(args.native_log, _native_log_payload(outcomes))

    for outcome in outcomes:
        if outcome.error_type is not None:
            print(
                f"hotpot xtb: record {outcome.index}: "
                f"{outcome.error_type}: {outcome.error_message}",
                file=sys.stderr,
            )
        elif outcome.status == "quality-failed":
            print(
                f"hotpot xtb: record {outcome.index}: post-check "
                f"profile {args.post_check!r} failed; the structure was emitted",
                file=sys.stderr,
            )
    return 0 if all(outcome.status == "ok" for outcome in outcomes) else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run an independent official GFN-FF or GFN-xTB node"
    )
    add_arguments(parser)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    return run(build_parser().parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
