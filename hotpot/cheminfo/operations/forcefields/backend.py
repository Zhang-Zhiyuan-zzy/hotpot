"""Native Open Babel force-field and coordinate-build primitives."""

from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import NoReturn, Optional

from hotpot.cheminfo.adapters.openbabel.forcefields import build as build_molecule
from hotpot.cheminfo.adapters.openbabel.forcefields import single_optimize as _native_single_optimize
from hotpot.cheminfo.adapters.openbabel.forcefields import (
    _native_module, _native_molecule_data, _seed_openbabel_random, _single_optimize_data,
)
from .contracts import (
    ForceFieldError,
    ForceFieldSetupError,
    ForceFieldSetupReport,
    ForceFieldSetupStage,
)


__all__ = ()


_SUPPORTED_FORCEFIELDS = frozenset(
    {"UFF", "MMFF94", "MMFF94s", "GAFF", "Ghemical"}
)


@dataclass(frozen=True)
class _CandidateOptimizationResult:
    energy: float
    energy_unit: str
    exploded: bool


_WORKER_LIFECYCLE_LOCK = threading.Lock()
_WORKER_EXIT_GRACE_SECONDS = 30.0


def _resolve_complex_forcefield(requested: Optional[str]) -> str:
    """Resolve every currently supported complex request to UFF."""
    if requested is not None and requested not in _SUPPORTED_FORCEFIELDS:
        raise ValueError(f"Unsupported force field: {requested!r}")
    return "UFF"


def _resolve_organic_forcefield(requested: Optional[str]) -> str:
    """Resolve an omitted organic force field without changing explicit choices."""
    effective = requested or "MMFF94s"
    if effective not in _SUPPORTED_FORCEFIELDS:
        raise ValueError(f"Unsupported force field: {effective!r}")
    return effective


def _raise_forcefield_setup_error(
    error: BaseException,
    *,
    requested_forcefield: Optional[str],
    effective_forcefield: str,
    stage: ForceFieldSetupStage,
) -> NoReturn:
    """Translate one structured native setup failure into the public error."""
    raise ForceFieldSetupError(
        str(error),
        ForceFieldSetupReport(
            requested_forcefield,
            effective_forcefield,
            stage,
        ),
    ) from error


def _single_ob_optimization(
    mol: "Molecule", forcefield: str, steps: int, *, selection=None
) -> _CandidateOptimizationResult:
    """Run one native segment against the complete or selected task graph."""
    from hotpot.cheminfo.structure._commit import _FactTransaction
    from hotpot.cheminfo.structure._fields import ATOM_SOURCE_CODES
    from hotpot.cheminfo.structure._identity import _capture_source, _check_source
    from hotpot.cheminfo.structure._selection import _check_topology_selection

    if selection is not None:
        with mol._mutation_lock:
            data = _native_molecule_data(mol, selection=selection)
            source = _capture_source(mol, ("bonding_inputs", "geometry", "cell"))
    native = _native_module()
    try:
        if selection is None:
            result = _native_single_optimize(mol, forcefield, steps)
        else:
            result = _single_optimize_data(data, forcefield, steps)
    except native.ForceFieldSetupError as error:
        _raise_forcefield_setup_error(
            error,
            requested_forcefield=forcefield,
            effective_forcefield=forcefield,
            stage=error.stage,
        )
    if selection is not None:
        with _FactTransaction(mol, producer=ATOM_SOURCE_CODES["forcefield"]):
            _check_topology_selection(mol, selection)
            _check_source(mol, source)
            mol.coordinates = result.coordinates
    return _CandidateOptimizationResult(
        energy=result.energy,
        energy_unit=result.energy_unit,
        exploded=result.exploded,
    )


def _ob_build(mol: "Molecule") -> None:
    """Build coordinates in the native backend under the seed lifecycle lock."""
    with _WORKER_LIFECYCLE_LOCK:
        if not build_molecule(mol).succeeded:
            error = ForceFieldError("Open Babel could not build initial 3D coordinates")
            error.sample_failure_code = "FF-BUILD-UNAVAILABLE"
            raise error
