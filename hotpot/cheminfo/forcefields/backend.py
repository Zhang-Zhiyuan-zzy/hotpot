"""Native Open Babel force-field and coordinate-build primitives."""

from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import NoReturn, Optional, TYPE_CHECKING

from ..obWrappers import build as build_molecule
from ..obWrappers.forcefield import _single_optimize as _native_single_optimize
from ..obWrappers.native import _native_module
from .contracts import ForceFieldError, ForceFieldSetupError, ForceFieldSetupReport


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
    stage: str,
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
    mol: "Molecule", forcefield: str, steps: int
) -> _CandidateOptimizationResult:
    """Run one native steepest-descent segment on ``mol``."""
    native = _native_module()
    try:
        result = _native_single_optimize(mol, forcefield, steps)
    except native.ForceFieldSetupError as error:
        _raise_forcefield_setup_error(
            error,
            requested_forcefield=forcefield,
            effective_forcefield=forcefield,
            stage=error.stage,
        )
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
