"""Controlled-pipeline adapter for independent GFN-FF and GFN-xTB nodes."""

from __future__ import annotations

import argparse
import math
import os
from dataclasses import dataclass, fields, is_dataclass
from enum import Enum
from numbers import Integral, Real
from pathlib import Path
from typing import Mapping, Optional, Tuple

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

from . import workflow
from .contracts import (
    GFNXTBMethod,
    XTBError,
    XTBExecutionError,
    XTBMethod,
    XTBResultError,
    XTBRunReport,
    XTBTask,
)


__all__ = ("STAGE", "get_stage")


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


class _StageArgumentParser(argparse.ArgumentParser):
    """Report xTB syntax failures through the pipeline definition contract."""

    def error(self, message: str) -> None:
        raise PipelineDefinitionError(f"xtb stage: {message}")


@dataclass(frozen=True)
class _NamedChargeEstimator:
    model: str

    def infer(self, mol: Molecule) -> ChargeInferenceResult:
        return infer_charge(mol, model=self.model)


@dataclass(frozen=True)
class _ResolvedChargeEstimator:
    result: ChargeInferenceResult

    def infer(self, mol: Molecule) -> ChargeInferenceResult:
        return self.result


@dataclass(frozen=True)
class _XTBStageOptions:
    method: XTBMethod
    task: XTBTask
    charge: Optional[int]
    unpaired_electrons: Optional[int]
    charge_model: str
    executable: Optional[str]
    threads: Optional[int]
    post_check: str


@dataclass(frozen=True)
class _PreparedXTBStage(PreparedMolecularStage):
    options: _XTBStageOptions

    def execute(
        self,
        payload: MolecularPayload,
        context: StageContext,
    ) -> StageResult:
        if not payload.records:
            raise StageExecutionError("xtb stage requires at least one input molecule")

        output_records = []
        reports = []
        quality_reports = []
        native_logs = []
        quality_failed = False
        for index, record in enumerate(payload.records):
            try:
                report, state = _run_record(
                    record,
                    self.options,
                    context.stage_directory / "native",
                )
                quality_report = _post_check(record.molecule, self.options.post_check)
            except (XTBExecutionError, XTBResultError) as error:
                native_logs.extend(_native_log_sections(index, error.report))
                _write_native_log(context.stage_directory, native_logs)
                raise StageExecutionError(
                    f"xtb stage record {index} failed: {error}"
                ) from error
            except (XTBError, ElectronicStateError, ValueError) as error:
                native_logs.append(
                    f"=== record {index} failure ===\n{type(error).__name__}: {error}\n"
                )
                _write_native_log(context.stage_directory, native_logs)
                raise StageExecutionError(
                    f"xtb stage record {index} failed: {error}"
                ) from error

            if quality_report is not None:
                quality_reports.append(_json_value(quality_report))
                quality_failed = quality_failed or not quality_report.passed
            reports.append(
                _report_summary(
                    report,
                    state,
                    self.options,
                    context.stage_directory,
                )
            )
            native_logs.extend(_native_log_sections(index, report))
            output_records.append(MolecularRecord(record.molecule, state))

        native_log_path = context.stage_directory / "native.log"
        _write_native_log(context.stage_directory, native_logs)
        artifacts = _artifacts_under(context.stage_directory / "native", context)
        artifacts += (artifact_from_file(context.stage_directory, native_log_path),)
        xtb_report = reports[0] if len(reports) == 1 else tuple(reports)
        stage_report: dict[str, JSONValue] = {"xtb": xtb_report}
        if quality_reports:
            stage_report["post_check"] = tuple(quality_reports)
        status = (
            StageStatus.QUALITY_FAILED
            if quality_failed
            else StageStatus.SUCCEEDED
        )
        stderr = (
            "One or more xTB structures failed the requested post-check profile."
            if quality_failed
            else ""
        )
        return StageResult(
            status=status,
            payload=MolecularPayload(tuple(output_records)),
            artifacts=artifacts,
            report=stage_report,
            stderr=stderr,
        )


@dataclass(frozen=True)
class _XTBStage(MolecularStage):
    def prepare(self, spec: StageSpec) -> PreparedMolecularStage:
        if spec.name != "xtb":
            raise PipelineDefinitionError(
                f"xTB stage cannot prepare stage {spec.name!r}"
            )
        args = _stage_parser().parse_args(spec.argv)
        if args.method == "gfnff" and args.unpaired_electrons is not None:
            raise PipelineDefinitionError(
                "xtb stage: GFN-FF does not accept --unpaired-electrons"
            )
        return _PreparedXTBStage(
            _XTBStageOptions(
                method=_METHODS[args.method],
                task=_TASKS[args.task],
                charge=args.charge,
                unpaired_electrons=args.unpaired_electrons,
                charge_model=args.charge_model,
                executable=args.xtb_executable,
                threads=args.threads,
                post_check=args.post_check,
            )
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


def _stage_parser() -> _StageArgumentParser:
    parser = _StageArgumentParser(add_help=False)
    parser.add_argument(
        "--method",
        choices=tuple(_METHODS),
        default="gfn2",
    )
    parser.add_argument(
        "--task",
        choices=tuple(_TASKS),
        default="optimize",
    )
    parser.add_argument("--charge", type=int)
    parser.add_argument("--unpaired-electrons", type=_nonnegative_int)
    parser.add_argument(
        "--charge-model",
        choices=("valence", "valence-constrained", "preserve"),
        default="valence",
    )
    parser.add_argument("--xtb-executable")
    parser.add_argument("--threads", type=_positive_int)
    parser.add_argument(
        "--post-check",
        choices=("off", "basic", "standard", "strict"),
        default="off",
    )
    return parser


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


def _resolved_state(
    record: MolecularRecord,
    options: _XTBStageOptions,
) -> ElectronicState:
    if (
        record.electronic_state is not None
        and options.charge is None
        and options.unpaired_electrons is None
    ):
        return record.electronic_state

    carried_state = record.electronic_state
    charge = options.charge
    unpaired_electrons = options.unpaired_electrons
    if carried_state is not None:
        if charge is None:
            charge = carried_state.charge
        if unpaired_electrons is None:
            unpaired_electrons = carried_state.unpaired_electrons
    estimator = (
        None if charge is not None else _NamedChargeEstimator(options.charge_model)
    )
    return resolve_electronic_state(
        record.molecule,
        charge=charge,
        unpaired_electrons=unpaired_electrons,
        charge_estimator=estimator,
    )


def _run_record(
    record: MolecularRecord,
    options: _XTBStageOptions,
    work_directory: Path,
) -> Tuple[XTBRunReport, Optional[ElectronicState]]:
    common = {
        "task": options.task,
        "executable": options.executable,
        "environment": _environment(options.threads),
        "work_directory": work_directory,
        "keep_work_directory": True,
    }
    if options.method is XTBMethod.GFNFF:
        if record.electronic_state is not None and options.charge is None:
            state = record.electronic_state
            report = workflow.run_gfnff(
                record.molecule,
                charge=state.charge,
                **common,
            )
        elif options.charge is not None:
            state = _resolved_state(record, options)
            report = workflow.run_gfnff(
                record.molecule,
                charge=options.charge,
                **common,
            )
        else:
            charge_result = infer_charge(
                record.molecule,
                model=options.charge_model,
            )
            state = resolve_electronic_state(
                record.molecule,
                charge_estimator=_ResolvedChargeEstimator(charge_result),
            )
            report = workflow.run_gfnff(
                record.molecule,
                charge_state=charge_result,
                **common,
            )
        return report, state

    state = _resolved_state(record, options)
    report = workflow.run_gfn_xtb(
        record.molecule,
        method=_GFN_XTB_METHODS[options.method.value],
        state=state,
        **common,
    )
    return report, state


def _post_check(mol: Molecule, level: str) -> Optional[object]:
    if level == "off":
        return None
    from hotpot.cheminfo.forcefields import evaluate_structure_acceptance

    return evaluate_structure_acceptance(mol, level=level)


def _report_text(report: object, field: str) -> str:
    return str(getattr(report, field, ""))


def _native_log_sections(index: int, report: object) -> Tuple[str, str]:
    return (
        f"=== record {index} stdout ===\n{_report_text(report, 'stdout').rstrip()}\n",
        f"=== record {index} stderr ===\n{_report_text(report, 'stderr').rstrip()}\n",
    )


def _write_native_log(stage_directory: Path, sections: list[str]) -> None:
    (stage_directory / "native.log").write_text(
        "".join(sections),
        encoding="utf-8",
    )


def _report_summary(
    report: XTBRunReport,
    state: Optional[ElectronicState],
    options: _XTBStageOptions,
    stage_directory: Path,
) -> Mapping[str, JSONValue]:
    summary: dict[str, JSONValue] = {
        "requested_method": options.method.value,
        "task": options.task.value,
        "electronic_state": _json_value(state),
        "energy_hartree": _json_value(getattr(report, "energy_hartree", None)),
        "gradient_norm": _json_value(getattr(report, "gradient_norm", None)),
        "process_succeeded": _json_value(
            getattr(report, "process_succeeded", None)
        ),
        "converged": _json_value(getattr(report, "converged", None)),
        "coordinates_committed": _json_value(
            getattr(report, "coordinates_committed", None)
        ),
    }
    if isinstance(report, XTBRunReport):
        summary["effective_method"] = report.effective_method.value
        summary["charge"] = report.charge
        summary["unpaired_electrons"] = report.unpaired_electrons
        summary["backend"] = {
            "version": report.backend_info.version,
            "revision": report.backend_info.revision,
            "executable_sha256": report.backend_info.executable_sha256,
        }
        summary["native_artifacts"] = tuple(
            {
                "name": artifact.name,
                "relative_path": artifact.path.absolute()
                .relative_to(stage_directory.absolute())
                .as_posix(),
                "sha256": artifact.sha256,
                "size_bytes": artifact.size_bytes,
            }
            for artifact in report.artifacts.values()
        )
    return summary


def _json_value(value: object) -> JSONValue:
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
        return tuple(_json_value(item) for item in value)
    raise TypeError(f"Cannot serialize xTB value {type(value)!r}")


def _artifacts_under(path: Path, context: StageContext) -> Tuple[Artifact, ...]:
    if not path.exists():
        return ()
    return tuple(
        artifact_from_file(context.stage_directory, item)
        for item in sorted(path.rglob("*"))
        if item.is_file()
    )


STAGE = _XTBStage()


def get_stage() -> MolecularStage:
    """Return the registered xTB molecular stage."""
    return STAGE
