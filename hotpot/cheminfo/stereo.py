"""Hotpot-owned stereochemistry transport and semantic contracts."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Mapping

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    from .core import Molecule


__all__ = (
    "AtomTransportRecord",
    "BondTransportRecord",
    "HydrogenTransportState",
    "StereoAlignmentReport",
    "StereoAnchor",
    "StereoAnchorKind",
    "StereoAssignmentReport",
    "StereoCarrier",
    "StereoCarrierKind",
    "StereoGroupKind",
    "StereoGroupRef",
    "StereoIssueCode",
    "StereoPayload",
    "StereoRecord",
    "StereoType",
    "StereoValidationIssue",
    "StereoValidationReport",
    "align_stereochemistry",
    "assign_stereochemistry",
    "get_stereo_payload",
    "set_stereo_payload",
    "validate_stereochemistry",
)


class StereoAnchorKind(Enum):
    """Graph object that owns a transported stereo record."""

    ATOM = "atom"
    BOND = "bond"


class StereoCarrierKind(Enum):
    """Reference kind participating in an ordered stereo descriptor."""

    ATOM = "atom"
    IMPLICIT_H = "implicit_h"
    LONE_PAIR = "lone_pair"
    VACANCY = "vacancy"


class StereoGroupKind(Enum):
    """Enhanced-stereo group relationship."""

    ABS = "abs"
    AND = "and"
    OR = "or"


@dataclass(frozen=True)
class StereoGroupRef:
    """Persistent lineage and source identity for an enhanced-stereo group."""

    scope: str
    kind: StereoGroupKind
    id: int


class StereoType(Enum):
    """Hotpot semantic families interpreted by Level 2."""

    TETRAHEDRAL = "tetrahedral"
    DOUBLE_BOND = "double_bond"
    ALLENE = "allene"
    ATROPISOMERIC = "atropisomeric"
    SQUARE_PLANAR = "square_planar"
    TRIGONAL_BIPYRAMIDAL = "trigonal_bipyramidal"
    SQUARE_PYRAMIDAL = "square_pyramidal"
    OCTAHEDRAL = "octahedral"
    UNKNOWN = "unknown"


class StereoIssueCode(Enum):
    """Stable classification for Level-2 validation diagnostics."""

    INVALID_ANCHOR = "invalid_anchor"
    INVALID_CARRIER = "invalid_carrier"
    INVALID_DESCRIPTOR = "invalid_descriptor"
    NON_STEREOGENIC = "non_stereogenic"
    DEGENERATE_GEOMETRY = "degenerate_geometry"
    COORDINATE_CONTRADICTION = "coordinate_contradiction"
    INCOMPLETE_MAPPING = "incomplete_mapping"
    UNSUPPORTED = "unsupported"


@dataclass(frozen=True)
class StereoCarrier:
    """One ordered atom or non-atom stereo reference."""

    kind: StereoCarrierKind
    atom: int | None = None


@dataclass(frozen=True)
class StereoAnchor:
    """Stable atom or ordered bond endpoint anchor."""

    kind: StereoAnchorKind
    atom_indices: tuple[int, ...]


@dataclass(frozen=True)
class StereoRecord:
    """Uninterpreted stereo data from a named source namespace."""

    anchor: StereoAnchor
    carriers: tuple[StereoCarrier, ...]
    descriptor: str
    group: StereoGroupRef | None
    source_namespace: str
    source_code: str


@dataclass(frozen=True)
class HydrogenTransportState:
    """One source or current representation of atom-local hydrogen facts."""

    implicit_hydrogens: int | None
    bracket_explicit_hydrogens: int | None
    no_implicit: bool | None


@dataclass(frozen=True)
class AtomTransportRecord:
    """Current atom facts plus an immutable source hydrogen ledger."""

    atom: int
    isotope: int | None
    radical_electrons: int | None
    current_hydrogens: HydrogenTransportState | None
    source_hydrogens: HydrogenTransportState | None
    source_namespace: str | None


@dataclass(frozen=True)
class BondTransportRecord:
    """Display-direction facts independent of chemical bond kind."""

    atom_indices: tuple[int, int]
    display_direction: str | None


@dataclass(frozen=True)
class StereoPayload:
    """Versioned lossless Level-1 transport payload."""

    schema_version: int
    records: tuple[StereoRecord, ...]
    atom_facts: tuple[AtomTransportRecord, ...]
    bond_facts: tuple[BondTransportRecord, ...]


@dataclass(frozen=True)
class StereoValidationIssue:
    """One deterministic Level-2 diagnostic."""

    code: StereoIssueCode
    message: str
    record_index: int | None = None
    atom_indices: tuple[int, ...] = ()


@dataclass(frozen=True)
class StereoValidationReport:
    """Validation result for stored and optional spatial stereo facts."""

    is_valid: bool
    issues: tuple[StereoValidationIssue, ...]


@dataclass(frozen=True)
class StereoAssignmentReport:
    """Assigned payload and facts that could not be resolved."""

    payload: StereoPayload
    unresolved: tuple[StereoValidationIssue, ...]


@dataclass(frozen=True)
class StereoAlignmentReport:
    """Stereo equivalence after applying an explicit atom mapping."""

    is_equivalent: bool
    aligned_payload: StereoPayload
    issues: tuple[StereoValidationIssue, ...]


def get_stereo_payload(mol: Molecule) -> StereoPayload:
    """Return the exact Level-1 payload stored by a molecule."""

    raise NotImplementedError("Level-1 stereo storage is implemented in Phase 2")


def set_stereo_payload(mol: Molecule, payload: StereoPayload) -> None:
    """Store Level-1 facts without chemical interpretation."""

    raise NotImplementedError("Level-1 stereo storage is implemented in Phase 2")


def validate_stereochemistry(
    mol: Molecule,
    *,
    coordinates: NDArray[np.float64] | None = None,
) -> StereoValidationReport:
    """Validate representation and optional coordinate consistency."""

    raise NotImplementedError("Level-2 stereo semantics are implemented in Phase 7")


def assign_stereochemistry(
    mol: Molecule,
    *,
    coordinates: NDArray[np.float64] | None = None,
) -> StereoAssignmentReport:
    """Assign Hotpot stereo descriptors from topology and coordinates."""

    raise NotImplementedError("Level-2 stereo semantics are implemented in Phase 7")


def align_stereochemistry(
    reference: Molecule,
    target: Molecule,
    atom_mapping: Mapping[int, int],
) -> StereoAlignmentReport:
    """Compare two stereo payloads under an explicit atom mapping."""

    raise NotImplementedError("Level-2 stereo semantics are implemented in Phase 7")
