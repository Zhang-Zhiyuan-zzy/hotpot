"""Build xTB argv, invoke the backend, and validate native artifacts."""

from __future__ import annotations

from pathlib import Path
from types import MappingProxyType
from typing import Dict, Tuple

from hotpot.plugins._harness import (
    NativeProcessResult,
    ProcessRequest,
    build_process_provenance,
    run_process,
)

from .contracts import (
    XTBExecutionError,
    XTBArtifact,
    XTBInputError,
    XTBMethod,
    XTBRequest,
    XTBResultError,
    XTBRunReport,
    XTBTask,
)


__all__ = ["run_xtb"]


_METHOD_ARGUMENTS = {
    XTBMethod.GFNFF: ("--gfnff",),
    XTBMethod.GFN0_XTB: ("--gfn", "0"),
    XTBMethod.GFN1_XTB: ("--gfn", "1"),
    XTBMethod.GFN2_XTB: ("--gfn", "2"),
}
_TASK_ARGUMENTS = {
    XTBTask.SINGLEPOINT: "--sp",
    XTBTask.OPTIMIZE: "--opt",
}
_RESULT_ARTIFACT = {
    XTBMethod.GFNFF: "gfnff_lists.json",
    XTBMethod.GFN0_XTB: "xtbout.json",
    XTBMethod.GFN1_XTB: "xtbout.json",
    XTBMethod.GFN2_XTB: "xtbout.json",
}


def _build_argv(request: XTBRequest) -> Tuple[str, ...]:
    if request.method is XTBMethod.GFNFF and request.unpaired_electrons is not None:
        raise XTBInputError("GFN-FF does not accept an unpaired-electron option")
    if request.method is not XTBMethod.GFNFF and request.unpaired_electrons is None:
        raise XTBInputError("GFN-xTB requires an explicit unpaired-electron count")

    argv = (
        str(request.backend_info.executable),
        str(request.input_path.resolve()),
        *_METHOD_ARGUMENTS[request.method],
        _TASK_ARGUMENTS[request.task],
        "--json",
        "--chrg",
        str(request.charge),
    )
    if request.method is not XTBMethod.GFNFF:
        argv = (*argv, "--uhf", str(request.unpaired_electrons))
    return argv


def _collect_artifacts(work_directory: Path) -> Dict[str, Path]:
    return {
        path.name: path.resolve()
        for path in sorted(work_directory.iterdir(), key=lambda item: item.name)
        if path.is_file()
    }


def _required_artifacts(request: XTBRequest) -> Tuple[str, ...]:
    required = (_RESULT_ARTIFACT[request.method],)
    if request.task is XTBTask.OPTIMIZE:
        return (*required, ".xtboptok", "xtbopt.xyz", "xtbopt.log")
    return required


def _build_report(
    request: XTBRequest,
    process_result: NativeProcessResult,
    artifacts: Dict[str, Path],
) -> XTBRunReport:
    artifact_paths = tuple(artifacts.values())
    provenance = build_process_provenance(
        executable=request.backend_info.executable,
        result=process_result,
        artifacts=artifact_paths,
    )
    artifact_records = {
        artifact.path.name: XTBArtifact(
            name=artifact.path.name,
            path=artifact.path,
            sha256=artifact.sha256,
            size_bytes=artifact.path.stat().st_size,
        )
        for artifact in provenance.artifacts
    }
    return XTBRunReport(
        backend_info=request.backend_info,
        requested_method=request.method,
        effective_method=request.method,
        task=request.task,
        charge=request.charge,
        unpaired_electrons=request.unpaired_electrons,
        work_directory=request.work_directory.resolve(),
        argv=process_result.argv,
        return_code=process_result.return_code,
        stdout=process_result.stdout,
        stderr=process_result.stderr,
        elapsed_seconds=process_result.elapsed_seconds,
        process_succeeded=process_result.return_code == 0,
        converged=(
            process_result.return_code == 0
            and all(name in artifacts for name in _required_artifacts(request))
        ),
        artifacts=MappingProxyType(artifact_records),
        provenance=provenance,
    )


def run_xtb(request: XTBRequest) -> XTBRunReport:
    """Run xTB in an existing workspace and require method-specific evidence."""

    input_path = request.input_path.resolve()
    work_directory = request.work_directory.resolve()
    if not input_path.is_file():
        raise XTBInputError(f"xTB input file does not exist: {input_path}")
    if not work_directory.is_dir():
        raise XTBInputError(f"xTB work directory does not exist: {work_directory}")

    process_result = run_process(
        ProcessRequest(
            argv=_build_argv(request),
            cwd=work_directory,
            env=dict(request.environment),
            timeout_seconds=request.timeout_seconds,
        )
    )
    artifacts = _collect_artifacts(work_directory)
    report = _build_report(request, process_result, artifacts)

    if process_result.return_code != 0:
        raise XTBExecutionError(
            f"xTB exited with code {process_result.return_code}",
            report,
        )

    missing_artifacts = tuple(
        name for name in _required_artifacts(request) if name not in artifacts
    )
    if missing_artifacts:
        raise XTBResultError(
            f"xTB did not produce required artifacts: {missing_artifacts!r}",
            report,
        )
    return report
