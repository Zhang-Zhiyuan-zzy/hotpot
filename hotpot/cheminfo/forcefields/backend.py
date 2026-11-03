"""Native Open Babel force-field and coordinate-build primitives."""

from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import NoReturn, Optional, TYPE_CHECKING

from ..obWrappers import build as build_molecule
from ..obWrappers import single_optimize as _native_single_optimize
from ..obWrappers.native import _native_module
from .contracts import (
    ForceFieldError,
    ForceFieldSetupError,
    ForceFieldSetupReport,
    ForceFieldSetupStage,
)


if TYPE_CHECKING:
    from ..core import Molecule


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


def _seed_openbabel_random(seed: int) -> None:
    """Seed the version-matched native Open Babel random generators."""
    _native_module().seed_random(seed)


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
    from ..operations.topology_queries import _project_molecule
    from ..structure._commit import _FactTransaction
    from ..structure._fields import ATOM_SOURCE_CODES
    from ..structure._identity import _capture_source, _check_source
    from ..structure._selection import _check_topology_selection

    target = mol
    if selection is not None:
        with mol._mutation_lock:
            _check_topology_selection(mol, selection)
            source = _capture_source(mol, ("bonding_inputs", "geometry", "cell"))
            target = _project_molecule(mol, type(mol), selection=selection)
    native = _native_module()
    try:
        result = _native_single_optimize(target, forcefield, steps)
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
            mol.coordinates = target.coordinates
    return _CandidateOptimizationResult(
        energy=result.energy,
        energy_unit=result.energy_unit,
        exploded=result.exploded,
    )


def _ob_build(mol: "Molecule") -> None:
    """Build coordinates in the native backend under the seed lifecycle lock."""
    with _WORKER_LIFECYCLE_LOCK:
        if not build_molecule(mol).succeeded:
            raise ForceFieldError("Open Babel could not build initial 3D coordinates")
