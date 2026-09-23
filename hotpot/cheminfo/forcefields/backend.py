"""Serialized Open Babel force-field and coordinate-build primitives."""

from __future__ import annotations

import os
import threading
from dataclasses import dataclass
from functools import wraps
from typing import Callable, Optional, TYPE_CHECKING, TypeVar, cast

from openbabel import openbabel as ob

from ..obconvert import extract_obmol_coordinates, mol2obmol
from .contracts import ForceFieldError, ForceFieldSetupError, ForceFieldSetupReport


if TYPE_CHECKING:
    from ..core import Molecule


__all__ = ()


CallableT = TypeVar("CallableT", bound=Callable[..., object])


_SUPPORTED_FORCEFIELDS = frozenset({"UFF", "MMFF94", "MMFF94s", "GAFF", "Ghemical"})


@dataclass(frozen=True)
class _CandidateOptimizationResult:
    energy: float
    energy_unit: str
    exploded: bool


_WORKER_LIFECYCLE_LOCK = threading.Lock()
_OPENBABEL_FORCEFIELD_LOCK = threading.RLock()
_WORKER_EXIT_GRACE_SECONDS = 30.0


def _serialized_forcefield_call(function: CallableT) -> CallableT:
    @wraps(function)
    def synchronized(*args: object, **kwargs: object) -> object:
        with _OPENBABEL_FORCEFIELD_LOCK:
            return function(*args, **kwargs)

    return cast(CallableT, synchronized)


def _serialized_builder_call(function: CallableT) -> CallableT:
    @wraps(function)
    def synchronized(*args: object, **kwargs: object) -> object:
        with _WORKER_LIFECYCLE_LOCK:
            with _OPENBABEL_FORCEFIELD_LOCK:
                return function(*args, **kwargs)

    return cast(CallableT, synchronized)


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


def _make_constraints(mol: "Molecule") -> ob.OBFFConstraints:
    """Return the intentionally empty force-field constraint adapter."""
    return ob.OBFFConstraints()


def _setup_forcefield_backend(
    backend: ob.OBForceField,
    mol: "Molecule",
    obmol: ob.OBMol,
    *,
    requested_forcefield: Optional[str],
    effective_forcefield: str,
) -> None:
    """Set up an Open Babel force field or raise structured diagnostics."""
    if backend.Setup(obmol, _make_constraints(mol)):
        return
    raise ForceFieldSetupError(
        f"Open Babel could not initialize force field {effective_forcefield!r}",
        ForceFieldSetupReport(
            requested_forcefield,
            effective_forcefield,
            "setup",
        ),
    )


def _energy_factor_to_kj(unit: str) -> float:
    normalized = unit.strip().lower().replace(" ", "")
    if normalized in {"kj/mol", "kjmol-1", "kjmol^-1"}:
        return 1.0
    if normalized in {"kcal/mol", "kcalmol-1", "kcalmol^-1"}:
        return 4.184
    raise ValueError(f"Unsupported Open Babel energy unit: {unit!r}")


def _forcefield_energy_in_kj(
    ob_forcefield: ob.OBForceField,
    calc_grad: bool = True,
) -> float:
    return float(ob_forcefield.Energy(calc_grad)) * _energy_factor_to_kj(
        ob_forcefield.GetUnit()
    )


@_serialized_forcefield_call
def _get_forcefield(name: str) -> ob.OBForceField:
    """Retrieve a force-field plugin guarded by the process-local FF lock."""
    backend = _find_forcefield_prototype(name)
    if backend is None:
        raise ForceFieldSetupError(
            f"Unknown Open Babel force field: {name!r}",
            ForceFieldSetupReport(name, name, "lookup"),
        )
    return backend


def _find_forcefield_prototype(name: str) -> Optional[ob.OBForceField]:
    return ob.OBForceField.FindType(name)


def _seed_openbabel_random(seed: int) -> None:
    """Seed the current Open Babel RNG before using ``OBBuilder``."""
    os.environ["OB_RANDOM_SEED"] = str(seed)


@_serialized_forcefield_call
def _single_ob_optimization(
    mol: "Molecule", forcefield: str, steps: int
) -> _CandidateOptimizationResult:
    backend = _get_forcefield(forcefield)
    backend.EnableCutOff(False)
    obmol, _ = mol2obmol(mol)
    _setup_forcefield_backend(
        backend,
        mol,
        obmol,
        requested_forcefield=forcefield,
        effective_forcefield=forcefield,
    )
    backend.SteepestDescent(steps)
    backend.GetCoordinates(obmol)
    mol.coordinates = extract_obmol_coordinates(obmol)
    energy = _forcefield_energy_in_kj(backend)
    return _CandidateOptimizationResult(
        energy=energy,
        energy_unit="kJ/mol",
        exploded=bool(backend.DetectExplosion()),
    )


# Low-level Open Babel build and optimization primitives.


@_serialized_builder_call
def _ob_build(mol: "Molecule") -> None:
    """Run OBBuilder directly on an internal working molecule."""
    builder = ob.OBBuilder()
    obmol, _ = mol2obmol(mol)
    if not builder.Build(obmol):
        raise ForceFieldError("Open Babel could not build initial 3D coordinates")
    mol.coordinates = extract_obmol_coordinates(obmol)
