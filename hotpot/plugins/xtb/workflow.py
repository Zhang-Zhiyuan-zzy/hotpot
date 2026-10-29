"""Independent public GFN-FF and GFN-xTB workflow operations."""

from __future__ import annotations

import os
from dataclasses import replace
from pathlib import Path
from typing import Mapping, Optional, Union

from hotpot.cheminfo.calculator.execution_selection import (
    BackendStateProjection,
    _projection_validation_findings,
)
from hotpot.cheminfo.calculator.state_resolver import extract_molecular_facts
from hotpot.cheminfo.core import Molecule
from hotpot.cheminfo.structure._commit import _FactTransaction
from hotpot.cheminfo.structure._fields import ATOM_SOURCE_CODES
from hotpot.cheminfo.structure._identity import _capture_source, _check_source
from hotpot.plugins._harness import isolated_workspace

from .adapter import (
    commit_xtb_coordinates,
    parse_xtb_artifacts,
    prepare_xtb_input,
)
from .backend import probe_xtb_backend
from .capabilities import validate_element_support
from .contracts import (
    GFNXTBMethod,
    XTBExecutionError,
    XTBInputError,
    XTBMethod,
    XTBRequest,
    XTBResultError,
    XTBRunReport,
    XTBTask,
)
from .runner import run_xtb
from .state_projection import XTBBackendContext


__all__ = ["run_gfn_xtb", "run_gfnff"]


_ExecutablePath = Union[str, Path]


def _process_environment(
    environment: Optional[Mapping[str, str]],
) -> Mapping[str, str]:
    return dict(os.environ if environment is None else environment)


def _workspace_parent(work_directory: Optional[Path]) -> Optional[Path]:
    if work_directory is None:
        return None
    parent = Path(work_directory).expanduser().resolve()
    parent.mkdir(parents=True, exist_ok=True)
    return parent


def _enrich_report(
    report: XTBRunReport,
    *,
    state: BackendStateProjection,
    workspace_retained: bool,
) -> XTBRunReport:
    return replace(
        report,
        scientific_state_identity=state.scientific_state_identity,
        facts_identity=state.facts_identity,
        projection_convention=state.projection_convention,
        workspace_retained=workspace_retained,
    )


def _validate_projection(
    mol: Molecule,
    state: BackendStateProjection,
    context: XTBBackendContext,
) -> None:
    facts = extract_molecular_facts(mol)
    findings = _projection_validation_findings(facts, context, state)
    if findings:
        raise XTBInputError(
            "; ".join(f"{finding.code}: {finding.message}" for finding in findings)
        )


def _execute_workflow(
    mol: Molecule,
    *,
    method: XTBMethod,
    task: XTBTask,
    state: BackendStateProjection,
    executable: Optional[_ExecutablePath],
    environment: Optional[Mapping[str, str]],
    timeout_seconds: Optional[float],
    work_directory: Optional[Path],
    keep_work_directory: bool,
) -> XTBRunReport:
    process_environment = _process_environment(environment)
    workspace_parent = _workspace_parent(work_directory)

    with isolated_workspace(
        workspace_parent,
        prefix="hotpot-xtb-",
        retain=keep_work_directory,
    ) as workspace:
        input_path = workspace / (
            "input.sdf" if method is XTBMethod.GFNFF else "input.xyz"
        )
        with mol._mutation_lock:
            source = _capture_source(mol, ("topology", "geometry", "forcefield_inputs"))
            expected_geometry = prepare_xtb_input(mol, input_path)
        backend_info = probe_xtb_backend(
            executable,
            environment=process_environment,
            timeout_seconds=timeout_seconds,
        )
        validate_element_support(
            backend_info,
            method,
            (atom.atomic_number for atom in mol.atoms),
        )
        context = XTBBackendContext(backend_info, method)
        with mol._mutation_lock:
            _check_source(mol, source)
            _validate_projection(mol, state, context)
        request = XTBRequest(
            backend_info=backend_info,
            method=method,
            task=task,
            input_path=input_path,
            work_directory=workspace,
            charge=state.charge,
            unpaired_electrons=state.uhf,
            environment=process_environment,
            timeout_seconds=timeout_seconds,
        )
        try:
            report = run_xtb(request)
        except XTBExecutionError as error:
            raise XTBExecutionError(
                str(error),
                _enrich_report(
                    error.report,
                    state=state,
                    workspace_retained=keep_work_directory,
                ),
            ) from error
        except XTBResultError as error:
            raise XTBResultError(
                str(error),
                _enrich_report(
                    error.report,
                    state=state,
                    workspace_retained=keep_work_directory,
                ),
            ) from error

        report = _enrich_report(
            report,
            state=state,
            workspace_retained=keep_work_directory,
        )
        try:
            result = parse_xtb_artifacts(report, expected_geometry)
        except XTBResultError as error:
            raise XTBResultError(str(error), report) from error

        coordinates_committed = False
        if task is XTBTask.OPTIMIZE:
            try:
                with _FactTransaction(mol, producer=ATOM_SOURCE_CODES["xtb"]):
                    _check_source(mol, source)
                    commit_xtb_coordinates(mol, result)
            except XTBInputError as error:
                raise XTBResultError(
                    f"Validated xTB coordinates could not be committed: {error}",
                    report,
                ) from error
            coordinates_committed = True
        return replace(
            report,
            energy_hartree=result.energy_hartree,
            gradient_norm=result.gradient_norm,
            atom_order_verified=result.atom_order_verified,
            coordinates_committed=coordinates_committed,
        )


def run_gfnff(
    mol: Molecule,
    *,
    state: BackendStateProjection,
    task: XTBTask = XTBTask.OPTIMIZE,
    executable: Optional[_ExecutablePath] = None,
    environment: Optional[Mapping[str, str]] = None,
    timeout_seconds: Optional[float] = None,
    work_directory: Optional[Path] = None,
    keep_work_directory: bool = False,
) -> XTBRunReport:
    """Run an independent official GFN-FF node on ``mol``."""
    return _execute_workflow(
        mol,
        method=XTBMethod.GFNFF,
        task=task,
        state=state,
        executable=executable,
        environment=environment,
        timeout_seconds=timeout_seconds,
        work_directory=work_directory,
        keep_work_directory=keep_work_directory,
    )


def run_gfn_xtb(
    mol: Molecule,
    *,
    state: BackendStateProjection,
    method: GFNXTBMethod = GFNXTBMethod.GFN2_XTB,
    task: XTBTask = XTBTask.OPTIMIZE,
    executable: Optional[_ExecutablePath] = None,
    environment: Optional[Mapping[str, str]] = None,
    timeout_seconds: Optional[float] = None,
    work_directory: Optional[Path] = None,
    keep_work_directory: bool = False,
) -> XTBRunReport:
    """Run an independent official GFN-xTB node on ``mol``."""
    return _execute_workflow(
        mol,
        method=XTBMethod(method.value),
        task=task,
        state=state,
        executable=executable,
        environment=environment,
        timeout_seconds=timeout_seconds,
        work_directory=work_directory,
        keep_work_directory=keep_work_directory,
    )
