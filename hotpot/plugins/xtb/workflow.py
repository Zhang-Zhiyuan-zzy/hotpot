"""Independent public GFN-FF and GFN-xTB workflow operations."""

from __future__ import annotations

import os
from dataclasses import replace
from pathlib import Path
from typing import Mapping, Optional, Union

from hotpot.cheminfo.calculator.electronic_state import (
    ChargeInferenceResult,
    ChargeInferenceSource,
    ElectronicState,
    SpinInferenceSource,
)
from hotpot.cheminfo.calculator.electronic_state.resolver import (
    resolve_electronic_state,
)
from hotpot.cheminfo.calculator.formal_charges import infer_charge
from hotpot.cheminfo.core import Molecule
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
    charge_source: ChargeInferenceSource,
    spin_source: Optional[SpinInferenceSource],
    state_assumptions: tuple[str, ...],
    fragment_charges: tuple[int, ...],
    workspace_retained: bool,
) -> XTBRunReport:
    return replace(
        report,
        charge_source=charge_source,
        spin_source=spin_source,
        state_assumptions=state_assumptions,
        fragment_charges=fragment_charges,
        workspace_retained=workspace_retained,
    )


def _execute_workflow(
    mol: Molecule,
    *,
    method: XTBMethod,
    task: XTBTask,
    charge: int,
    unpaired_electrons: Optional[int],
    charge_source: ChargeInferenceSource,
    spin_source: Optional[SpinInferenceSource],
    state_assumptions: tuple[str, ...],
    fragment_charges: tuple[int, ...],
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
        input_path = workspace / "input.xyz"
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
        request = XTBRequest(
            backend_info=backend_info,
            method=method,
            task=task,
            input_path=input_path,
            work_directory=workspace,
            charge=charge,
            unpaired_electrons=unpaired_electrons,
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
                    charge_source=charge_source,
                    spin_source=spin_source,
                    state_assumptions=state_assumptions,
                    fragment_charges=fragment_charges,
                    workspace_retained=keep_work_directory,
                ),
            ) from error
        except XTBResultError as error:
            raise XTBResultError(
                str(error),
                _enrich_report(
                    error.report,
                    charge_source=charge_source,
                    spin_source=spin_source,
                    state_assumptions=state_assumptions,
                    fragment_charges=fragment_charges,
                    workspace_retained=keep_work_directory,
                ),
            ) from error

        report = _enrich_report(
            report,
            charge_source=charge_source,
            spin_source=spin_source,
            state_assumptions=state_assumptions,
            fragment_charges=fragment_charges,
            workspace_retained=keep_work_directory,
        )
        try:
            result = parse_xtb_artifacts(report, expected_geometry)
        except XTBResultError as error:
            raise XTBResultError(str(error), report) from error

        coordinates_committed = False
        if task is XTBTask.OPTIMIZE:
            try:
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
    task: XTBTask = XTBTask.OPTIMIZE,
    charge_state: Optional[ChargeInferenceResult] = None,
    charge: Optional[int] = None,
    executable: Optional[_ExecutablePath] = None,
    environment: Optional[Mapping[str, str]] = None,
    timeout_seconds: Optional[float] = None,
    work_directory: Optional[Path] = None,
    keep_work_directory: bool = False,
) -> XTBRunReport:
    """Run an independent official GFN-FF node on ``mol``."""
    if charge_state is not None and charge is not None:
        raise XTBInputError(
            "charge_state and an explicit charge cannot be supplied together"
        )

    if charge_state is not None:
        resolved_charge = charge_state.total_charge
        charge_source = charge_state.source
        assumptions = charge_state.assumptions
        fragment_charges = tuple(
            fragment.charge for fragment in charge_state.fragments
        )
    elif charge is not None:
        resolved_charge = charge
        charge_source = ChargeInferenceSource.EXPLICIT
        assumptions = (f"Used explicit total charge {charge}.",)
        fragment_charges = ()
    else:
        inferred_charge = infer_charge(mol)
        resolved_charge = inferred_charge.total_charge
        charge_source = inferred_charge.source
        assumptions = inferred_charge.assumptions
        fragment_charges = tuple(
            fragment.charge for fragment in inferred_charge.fragments
        )

    return _execute_workflow(
        mol,
        method=XTBMethod.GFNFF,
        task=task,
        charge=resolved_charge,
        unpaired_electrons=None,
        charge_source=charge_source,
        spin_source=None,
        state_assumptions=assumptions,
        fragment_charges=fragment_charges,
        executable=executable,
        environment=environment,
        timeout_seconds=timeout_seconds,
        work_directory=work_directory,
        keep_work_directory=keep_work_directory,
    )


def run_gfn_xtb(
    mol: Molecule,
    *,
    method: GFNXTBMethod = GFNXTBMethod.GFN2_XTB,
    task: XTBTask = XTBTask.OPTIMIZE,
    state: Optional[ElectronicState] = None,
    charge: Optional[int] = None,
    unpaired_electrons: Optional[int] = None,
    executable: Optional[_ExecutablePath] = None,
    environment: Optional[Mapping[str, str]] = None,
    timeout_seconds: Optional[float] = None,
    work_directory: Optional[Path] = None,
    keep_work_directory: bool = False,
) -> XTBRunReport:
    """Run an independent official GFN-xTB node on ``mol``."""
    if state is not None and (
        charge is not None or unpaired_electrons is not None
    ):
        raise XTBInputError(
            "state and explicit charge or spin overrides cannot be mixed"
        )

    resolved_state = state or resolve_electronic_state(
        mol,
        charge=charge,
        unpaired_electrons=unpaired_electrons,
    )
    return _execute_workflow(
        mol,
        method=XTBMethod(method.value),
        task=task,
        charge=resolved_state.charge,
        unpaired_electrons=resolved_state.unpaired_electrons,
        charge_source=resolved_state.charge_source,
        spin_source=resolved_state.spin_source,
        state_assumptions=resolved_state.assumptions,
        fragment_charges=resolved_state.fragment_charges,
        executable=executable,
        environment=environment,
        timeout_seconds=timeout_seconds,
        work_directory=work_directory,
        keep_work_directory=keep_work_directory,
    )
