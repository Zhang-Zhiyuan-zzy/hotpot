"""Immutable contracts for charge and spin inference."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Protocol, TYPE_CHECKING

if TYPE_CHECKING:
    from ...core import Molecule

__all__ = [
    "AmbiguousHydrogenRepresentationError",
    "ChargeEstimator",
    "ChargeInferenceError",
    "ChargeInferenceResult",
    "ChargeInferenceSource",
    "ElectronicState",
    "ElectronicStateError",
    "FragmentCharge",
    "IncompleteExplicitAtomError",
    "SpinEstimator",
    "SpinInferenceResult",
    "SpinInferenceSource",
]


# Exceptions and finite state sets.


class ElectronicStateError(ValueError):
    """Base error for an electronic state that cannot be resolved."""


class ChargeInferenceError(ElectronicStateError):
    """Raised when classical integer charge inference has no valid result."""


class AmbiguousHydrogenRepresentationError(ChargeInferenceError):
    """Raised when explicit and stored implicit hydrogen facts conflict."""


class IncompleteExplicitAtomError(ElectronicStateError):
    """Raised when an execution state omits atoms represented implicitly."""


class ChargeInferenceSource(str, Enum):
    """Provenance of an inferred or authoritative integer charge."""

    VALENCE = "valence"
    VALENCE_CONSTRAINED = "valence-constrained"
    PRESERVED = "preserve"
    METAL_DEFAULT = "metal-default"
    METAL_RESOLVER = "metal-resolver"
    EXPLICIT = "explicit"


class SpinInferenceSource(str, Enum):
    """Provenance of an unpaired-electron count."""

    LOWEST_SPIN_PARITY = "lowest-spin-parity"
    EXPLICIT = "explicit"


# Immutable data contracts.


@dataclass(frozen=True)
class FragmentCharge:
    """Integer charge assigned to one deterministic molecular fragment."""

    atom_indices: tuple[int, ...]
    charge: int
    source: ChargeInferenceSource


@dataclass(frozen=True)
class ChargeInferenceResult:
    """Non-mutating atom, fragment and molecular formal-charge evidence."""

    atom_formal_charges: tuple[int, ...]
    fragments: tuple[FragmentCharge, ...]
    total_charge: int
    source: ChargeInferenceSource
    assumptions: tuple[str, ...]


@dataclass(frozen=True)
class SpinInferenceResult:
    """Electron-count evidence for one spin policy."""

    unpaired_electrons: int
    multiplicity: int
    electron_count: int
    source: SpinInferenceSource
    assumptions: tuple[str, ...]


@dataclass(frozen=True)
class ElectronicState:
    """Resolved charge and spin values supplied to a numerical backend."""

    charge: int
    unpaired_electrons: int
    multiplicity: int
    fragment_charges: tuple[int, ...]
    charge_source: ChargeInferenceSource
    spin_source: SpinInferenceSource
    assumptions: tuple[str, ...]


# Replaceable calculator interfaces.


class ChargeEstimator(Protocol):
    """Interface for a non-mutating molecular charge estimator."""

    def infer(self, mol: "Molecule") -> ChargeInferenceResult:
        """Infer atom, fragment and total integer charges."""


class SpinEstimator(Protocol):
    """Interface for an unpaired-electron estimator."""

    def infer(self, mol: "Molecule", charge: int) -> SpinInferenceResult:
        """Infer a spin state for a fixed total charge."""

