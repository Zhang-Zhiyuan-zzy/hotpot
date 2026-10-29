"""Hotpot-owned stereochemistry transport and semantic contracts."""

from __future__ import annotations

from collections.abc import Collection, Sequence
from dataclasses import dataclass, replace
from enum import Enum
from itertools import combinations, permutations, product
from math import factorial, isfinite, radians, sqrt
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
    "StereoDerivedDescriptor",
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
    AXIS = "axis"


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


class _Level2DescriptorFamily(Enum):
    """Stable Hotpot descriptor tokens for Level-2 algebra.

    These tokens describe reference-relative configuration identity.  They
    are deliberately independent of derived CIP labels.
    """

    TETRAHEDRAL = "TH"
    DOUBLE_BOND = "DB"
    AXIAL = "AX"
    SQUARE_PLANAR = "SP"
    TRIGONAL_BIPYRAMIDAL = "TBP"
    SQUARE_PYRAMIDAL = "SPY"
    OCTAHEDRAL = "OH"


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
    """Stable atom, ordered bond, or complete ordered axis-path anchor."""

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


def _bond_key(atom_indices: Collection[int]) -> frozenset[int]:
    """Return the representation-independent key for a bond's endpoints."""

    return frozenset(atom_indices)


def _axis_bond_keys(atom_indices: Collection[int]) -> tuple[frozenset[int], ...]:
    """Return consecutive constituent-bond keys for an ordered axis path."""

    atom_indices = tuple(atom_indices)
    return tuple(
        _bond_key(atom_indices[index:index + 2])
        for index in range(len(atom_indices) - 1)
    )


def _remap_stereo_payload(
    payload: StereoPayload,
    atom_mapping: Mapping[int, int],
    *,
    allowed_bonds: Collection[tuple[int, int]] | None = None,
) -> StereoPayload:
    """Remap Level-1 facts without interpreting or reordering them.

    ``atom_mapping`` maps source atom rows to destination atom rows.
    ``allowed_bonds`` contains destination endpoint pairs; their orientation is
    ignored only for the existence check. Stored bond endpoint order is kept.
    """

    allowed_bond_keys = (
        None
        if allowed_bonds is None
        else {_bond_key(atom_indices) for atom_indices in allowed_bonds}
    )

    records = []
    for record in payload.records:
        if any(atom not in atom_mapping for atom in record.anchor.atom_indices):
            continue

        anchor_indices = tuple(
            atom_mapping[atom] for atom in record.anchor.atom_indices
        )
        if allowed_bond_keys is not None:
            if (
                record.anchor.kind is StereoAnchorKind.BOND
                and _bond_key(anchor_indices) not in allowed_bond_keys
            ):
                continue
            if (
                record.anchor.kind is StereoAnchorKind.AXIS
                and any(
                    bond_key not in allowed_bond_keys
                    for bond_key in _axis_bond_keys(anchor_indices)
                )
            ):
                continue

        carriers = []
        keep_record = True
        for carrier in record.carriers:
            if carrier.kind is StereoCarrierKind.ATOM:
                if carrier.atom not in atom_mapping:
                    keep_record = False
                    break
                carriers.append(replace(carrier, atom=atom_mapping[carrier.atom]))
            else:
                carriers.append(carrier)

        if keep_record:
            records.append(
                replace(
                    record,
                    anchor=replace(record.anchor, atom_indices=anchor_indices),
                    carriers=tuple(carriers),
                )
            )

    atom_facts = tuple(
        replace(fact, atom=atom_mapping[fact.atom])
        for fact in payload.atom_facts
        if fact.atom in atom_mapping
    )

    bond_facts = []
    for fact in payload.bond_facts:
        if any(atom not in atom_mapping for atom in fact.atom_indices):
            continue
        atom_indices = tuple(atom_mapping[atom] for atom in fact.atom_indices)
        if (
            allowed_bond_keys is not None
            and _bond_key(atom_indices) not in allowed_bond_keys
        ):
            continue
        bond_facts.append(replace(fact, atom_indices=atom_indices))

    return StereoPayload(
        schema_version=payload.schema_version,
        records=tuple(records),
        atom_facts=atom_facts,
        bond_facts=tuple(bond_facts),
    )


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
class _Level2Descriptor:
    """Parsed descriptor in the ``hotpot.stereo/2`` namespace."""

    family: _Level2DescriptorFamily
    value: str | int | None


@dataclass(frozen=True)
class _Level2AlgebraResult:
    """Value-or-issues result used by the chemistry-independent algebra."""

    descriptor: _Level2Descriptor | None
    issues: tuple[StereoValidationIssue, ...]


_LEVEL2_DESCRIPTOR_PREFIX = "hotpot.stereo/2"
_PARITY_VALUES = ("even", "odd")
_ALKENE_VALUES = ("together", "opposite")
_GEOMETRY_RELATIVE_TOLERANCE = 1.0e-8
_COORDINATION_CARDINALITIES = {
    _Level2DescriptorFamily.SQUARE_PLANAR: 3,
    _Level2DescriptorFamily.TRIGONAL_BIPYRAMIDAL: 20,
    _Level2DescriptorFamily.SQUARE_PYRAMIDAL: 30,
    _Level2DescriptorFamily.OCTAHEDRAL: 30,
}
_FAMILY_CARRIER_COUNTS = {
    _Level2DescriptorFamily.TETRAHEDRAL: 4,
    _Level2DescriptorFamily.DOUBLE_BOND: 4,
    _Level2DescriptorFamily.AXIAL: 4,
    _Level2DescriptorFamily.SQUARE_PLANAR: 4,
    _Level2DescriptorFamily.TRIGONAL_BIPYRAMIDAL: 5,
    _Level2DescriptorFamily.SQUARE_PYRAMIDAL: 5,
    _Level2DescriptorFamily.OCTAHEDRAL: 6,
}
_COORDINATION_FAMILIES = frozenset(
    (
        _Level2DescriptorFamily.SQUARE_PLANAR,
        _Level2DescriptorFamily.TRIGONAL_BIPYRAMIDAL,
        _Level2DescriptorFamily.SQUARE_PYRAMIDAL,
        _Level2DescriptorFamily.OCTAHEDRAL,
    )
)
_ATOM_GEOMETRY_FAMILIES = frozenset(
    (_Level2DescriptorFamily.TETRAHEDRAL, *_COORDINATION_FAMILIES)
)


def _algebra_issue(
    code: StereoIssueCode,
    message: str,
    *,
    atom_indices: tuple[int, ...] = (),
) -> StereoValidationIssue:
    return StereoValidationIssue(
        code=code,
        message=message,
        atom_indices=atom_indices,
    )


def _parse_level2_descriptor_head(
    text: str,
) -> tuple[
    _Level2DescriptorFamily | None,
    str | None,
    tuple[StereoValidationIssue, ...],
]:
    """Parse the namespace, version, family, and raw value token once."""

    parts = text.split("/")
    if len(parts) != 4 or "/".join(parts[:2]) != _LEVEL2_DESCRIPTOR_PREFIX:
        return (
            None,
            None,
            (
                _algebra_issue(
                    StereoIssueCode.INVALID_DESCRIPTOR,
                    "descriptor must use grammar hotpot.stereo/2/<family>/<value>",
                ),
            ),
        )
    try:
        family = _Level2DescriptorFamily(parts[2])
    except ValueError:
        return (
            None,
            parts[3],
            (
                _algebra_issue(
                    StereoIssueCode.UNSUPPORTED,
                    f"unsupported Level-2 descriptor family: {parts[2]}",
                ),
            ),
        )
    return family, parts[3], ()


def _parse_level2_descriptor(text: str) -> _Level2AlgebraResult:
    """Parse the strict, versioned Hotpot Level-2 descriptor grammar."""

    family, value_token, head_issues = _parse_level2_descriptor_head(text)
    if head_issues:
        return _Level2AlgebraResult(None, head_issues)
    if family in (
        _Level2DescriptorFamily.TETRAHEDRAL,
        _Level2DescriptorFamily.AXIAL,
    ):
        if value_token not in _PARITY_VALUES:
            return _Level2AlgebraResult(
                None,
                (
                    _algebra_issue(
                        StereoIssueCode.INVALID_DESCRIPTOR,
                        f"{family.value} descriptor must be even or odd",
                    ),
                ),
            )
        return _Level2AlgebraResult(_Level2Descriptor(family, value_token), ())

    if family is _Level2DescriptorFamily.DOUBLE_BOND:
        if value_token not in _ALKENE_VALUES:
            return _Level2AlgebraResult(
                None,
                (
                    _algebra_issue(
                        StereoIssueCode.INVALID_DESCRIPTOR,
                        "DB descriptor must be together or opposite",
                    ),
                ),
            )
        return _Level2AlgebraResult(_Level2Descriptor(family, value_token), ())

    try:
        index = int(value_token)
    except ValueError:
        index = 0
    cardinality = _COORDINATION_CARDINALITIES[family]
    if str(index) != value_token or not 1 <= index <= cardinality:
        return _Level2AlgebraResult(
            None,
            (
                _algebra_issue(
                    StereoIssueCode.INVALID_DESCRIPTOR,
                    f"{family.value} descriptor index must be in 1..{cardinality}",
                ),
            ),
        )
    return _Level2AlgebraResult(_Level2Descriptor(family, index), ())


def _render_level2_descriptor(descriptor: _Level2Descriptor) -> str:
    """Render one parsed descriptor without toolkit-specific translation."""

    value = "unsupported" if descriptor.value is None else str(descriptor.value)
    return f"{_LEVEL2_DESCRIPTOR_PREFIX}/{descriptor.family.value}/{value}"


def _declares_gate7_coordination(text: str) -> bool:
    """Return whether a valid Level-2 head declares a coordination family."""

    family, _, _ = _parse_level2_descriptor_head(text)
    return family in _COORDINATION_FAMILIES


def _validate_stereo_carriers(
    carriers: Sequence[StereoCarrier],
    *,
    expected_count: int,
) -> tuple[StereoValidationIssue, ...]:
    """Validate carrier shape and identity without chemical interpretation."""

    issues = []
    if len(carriers) != expected_count:
        issues.append(
            _algebra_issue(
                StereoIssueCode.INVALID_CARRIER,
                f"expected {expected_count} ordered carriers, got {len(carriers)}",
            )
        )

    identities = []
    for position, carrier in enumerate(carriers):
        if carrier.kind is StereoCarrierKind.ATOM:
            if (
                not isinstance(carrier.atom, int)
                or isinstance(carrier.atom, bool)
                or carrier.atom < 0
            ):
                issues.append(
                    _algebra_issue(
                        StereoIssueCode.INVALID_CARRIER,
                        f"atom carrier at position {position} requires a "
                        "non-negative atom row",
                    )
                )
        elif carrier.atom is not None:
            issues.append(
                _algebra_issue(
                    StereoIssueCode.INVALID_CARRIER,
                    f"non-atom carrier at position {position} cannot carry an atom row",
                )
            )
        identities.append((carrier.kind, carrier.atom))

    seen = set()
    for position, identity in enumerate(identities):
        if identity in seen:
            issues.append(
                _algebra_issue(
                    StereoIssueCode.INVALID_CARRIER,
                    f"duplicate carrier identity at position {position}",
                )
            )
        seen.add(identity)
    return tuple(issues)


def _validate_stereo_anchor(
    anchor: StereoAnchor,
    *,
    family: _Level2DescriptorFamily,
) -> tuple[StereoValidationIssue, ...]:
    """Validate the graph identity and orientation carried by an anchor."""

    issues = []
    if family is _Level2DescriptorFamily.AXIAL:
        valid_shape = (
            anchor.kind is StereoAnchorKind.BOND
            and len(anchor.atom_indices) == 2
        ) or (
            anchor.kind is StereoAnchorKind.AXIS
            and len(anchor.atom_indices) >= 3
            and len(anchor.atom_indices) % 2 == 1
        )
        requirement = (
            "a bond anchor with 2 atom rows or an odd axis path of at least 3 rows"
        )
    else:
        bond_family = family is _Level2DescriptorFamily.DOUBLE_BOND
        expected_kind = (
            StereoAnchorKind.BOND if bond_family else StereoAnchorKind.ATOM
        )
        expected_count = 2 if bond_family else 1
        valid_shape = (
            anchor.kind is expected_kind
            and len(anchor.atom_indices) == expected_count
        )
        requirement = (
            f"a {expected_kind.value} anchor with {expected_count} atom row(s)"
        )
    if not valid_shape:
        issues.append(
            _algebra_issue(
                StereoIssueCode.INVALID_ANCHOR,
                f"{family.value} requires {requirement}",
            )
        )
    if any(
        not isinstance(atom, int) or isinstance(atom, bool) or atom < 0
        for atom in anchor.atom_indices
    ):
        issues.append(
            _algebra_issue(
                StereoIssueCode.INVALID_ANCHOR,
                "anchor atom rows must be non-negative integers",
            )
        )
    if (
        anchor.kind in (StereoAnchorKind.BOND, StereoAnchorKind.AXIS)
        and len(set(anchor.atom_indices)) != len(anchor.atom_indices)
    ):
        issues.append(
            _algebra_issue(
                StereoIssueCode.INVALID_ANCHOR,
                "bond or axis anchor atom rows must be distinct",
            )
        )
    return tuple(issues)


def _validated_permutation(permutation: Sequence[int]) -> tuple[int, ...]:
    permutation = tuple(permutation)
    if sorted(permutation) != list(range(len(permutation))):
        raise ValueError("permutation must contain each index exactly once")
    return permutation


def _permutation_parity(permutation: Sequence[int]) -> int:
    """Return 0 for an even permutation and 1 for an odd permutation."""

    permutation = _validated_permutation(permutation)
    return sum(
        left > right
        for index, left in enumerate(permutation)
        for right in permutation[index + 1 :]
    ) % 2


def _permutation_rank(permutation: Sequence[int]) -> int:
    """Return the exact zero-based lexicographic Lehmer rank."""

    permutation = _validated_permutation(permutation)
    remaining = list(range(len(permutation)))
    rank = 0
    for position, item in enumerate(permutation):
        offset = remaining.index(item)
        rank += offset * factorial(len(permutation) - position - 1)
        remaining.pop(offset)
    return rank


def _permutation_unrank(size: int, rank: int) -> tuple[int, ...]:
    """Invert ``_permutation_rank`` exactly for a fixed permutation size."""

    if size < 0 or not 0 <= rank < factorial(size):
        raise ValueError("rank is outside the permutation family")
    remaining = list(range(size))
    permutation = []
    for width in range(size, 0, -1):
        block = factorial(width - 1)
        offset, rank = divmod(rank, block)
        permutation.append(remaining.pop(offset))
    return tuple(permutation)


_SQUARE_PLANAR_ROTATIONS = (
    (0, 1, 2, 3),
    (1, 2, 3, 0),
    (2, 3, 0, 1),
    (3, 0, 1, 2),
    (0, 3, 2, 1),
    (2, 1, 0, 3),
    (1, 0, 3, 2),
    (3, 2, 1, 0),
)
_TRIGONAL_BIPYRAMIDAL_ROTATIONS = (
    (0, 1, 2, 3, 4),
    (0, 1, 3, 4, 2),
    (0, 1, 4, 2, 3),
    (1, 0, 2, 4, 3),
    (1, 0, 3, 2, 4),
    (1, 0, 4, 3, 2),
)
_SQUARE_PYRAMIDAL_ROTATIONS = (
    (0, 1, 2, 3, 4),
    (0, 2, 3, 4, 1),
    (0, 3, 4, 1, 2),
    (0, 4, 1, 2, 3),
)


def _permutation_sign(permutation: Sequence[int]) -> int:
    return -1 if _permutation_parity(permutation) else 1


def _octahedral_rotations() -> tuple[tuple[int, ...], ...]:
    """Enumerate the 24 orientation-preserving signed-axis rotations."""

    vertices = (
        (1, 0, 0),
        (-1, 0, 0),
        (0, 1, 0),
        (0, -1, 0),
        (0, 0, 1),
        (0, 0, -1),
    )
    vertex_index = {vertex: index for index, vertex in enumerate(vertices)}
    rotations = []
    for axes in permutations(range(3)):
        for signs in product((-1, 1), repeat=3):
            if _permutation_sign(axes) * signs[0] * signs[1] * signs[2] != 1:
                continue
            rotation = []
            for vertex in vertices:
                transformed = [0, 0, 0]
                for output_axis, input_axis in enumerate(axes):
                    transformed[output_axis] = signs[output_axis] * vertex[input_axis]
                rotation.append(vertex_index[tuple(transformed)])
            rotations.append(tuple(rotation))
    return tuple(sorted(rotations))


_OCTAHEDRAL_ROTATIONS = _octahedral_rotations()
_COORDINATION_ROTATIONS = {
    _Level2DescriptorFamily.SQUARE_PLANAR: _SQUARE_PLANAR_ROTATIONS,
    _Level2DescriptorFamily.TRIGONAL_BIPYRAMIDAL: (
        _TRIGONAL_BIPYRAMIDAL_ROTATIONS
    ),
    _Level2DescriptorFamily.SQUARE_PYRAMIDAL: _SQUARE_PYRAMIDAL_ROTATIONS,
    _Level2DescriptorFamily.OCTAHEDRAL: _OCTAHEDRAL_ROTATIONS,
}


def _coordination_representatives(
    size: int,
    rotations: Sequence[Sequence[int]],
) -> tuple[tuple[int, ...], ...]:
    unseen = set(permutations(range(size)))
    representatives = []
    while unseen:
        seed = min(unseen)
        orbit = {
            tuple(seed[position] for position in rotation)
            for rotation in rotations
        }
        representative = min(orbit)
        representatives.append(representative)
        unseen.difference_update(orbit)
    return tuple(sorted(representatives))


_COORDINATION_REPRESENTATIVES = {
    family: _coordination_representatives(
        _FAMILY_CARRIER_COUNTS[family],
        rotations,
    )
    for family, rotations in _COORDINATION_ROTATIONS.items()
}

_COORDINATION_IDEAL_DIRECTIONS = {
    _Level2DescriptorFamily.SQUARE_PLANAR: (
        (1.0, 0.0, 0.0),
        (0.0, 1.0, 0.0),
        (-1.0, 0.0, 0.0),
        (0.0, -1.0, 0.0),
    ),
    _Level2DescriptorFamily.TRIGONAL_BIPYRAMIDAL: (
        (0.0, 0.0, 1.0),
        (0.0, 0.0, -1.0),
        (1.0, 0.0, 0.0),
        (-0.5, sqrt(3.0) / 2.0, 0.0),
        (-0.5, -sqrt(3.0) / 2.0, 0.0),
    ),
    _Level2DescriptorFamily.SQUARE_PYRAMIDAL: (
        (0.0, 0.0, 1.0),
        (1.0, 0.0, 0.0),
        (0.0, 1.0, 0.0),
        (-1.0, 0.0, 0.0),
        (0.0, -1.0, 0.0),
    ),
    _Level2DescriptorFamily.OCTAHEDRAL: (
        (1.0, 0.0, 0.0),
        (-1.0, 0.0, 0.0),
        (0.0, 1.0, 0.0),
        (0.0, -1.0, 0.0),
        (0.0, 0.0, 1.0),
        (0.0, 0.0, -1.0),
    ),
}
_GATE7_ACCEPTANCE_RADIANS = radians(12.0)
_GATE7_SAME_FAMILY_MARGIN_RADIANS = radians(2.0)
_GATE7_FIVE_COORDINATE_MARGIN_RADIANS = radians(3.0)
_GATE7_FIVE_COORDINATE_FAMILIES = (
    _Level2DescriptorFamily.TRIGONAL_BIPYRAMIDAL,
    _Level2DescriptorFamily.SQUARE_PYRAMIDAL,
)


def _carrier_reordering(
    source: Sequence[StereoCarrier],
    reordered: Sequence[StereoCarrier],
) -> tuple[int, ...] | None:
    source_identities = tuple((carrier.kind, carrier.atom) for carrier in source)
    reordered_identities = tuple(
        (carrier.kind, carrier.atom) for carrier in reordered
    )
    if set(source_identities) != set(reordered_identities):
        return None
    return tuple(source_identities.index(identity) for identity in reordered_identities)


def _is_endpoint_pair_reordering(permutation: Sequence[int]) -> bool:
    first = frozenset(permutation[:2])
    second = frozenset(permutation[2:])
    return (first, second) in (
        (frozenset((0, 1)), frozenset((2, 3))),
        (frozenset((2, 3)), frozenset((0, 1))),
    )


def _endpoint_pair_reordering(
    source: Sequence[StereoCarrier],
    reordered: Sequence[StereoCarrier],
    *,
    endpoint_groups_reversed: bool,
) -> tuple[int, ...] | None:
    """Match carrier identities within endpoint scope, including virtual H."""

    if len(source) != 4 or len(reordered) != 4:
        return None
    reordering = []
    for reordered_group in range(2):
        source_group = (
            1 - reordered_group if endpoint_groups_reversed else reordered_group
        )
        source_start = 2 * source_group
        source_carriers = source[source_start : source_start + 2]
        for carrier in reordered[2 * reordered_group : 2 * reordered_group + 2]:
            identity = carrier.kind, carrier.atom
            positions = tuple(
                source_start + offset
                for offset, candidate in enumerate(source_carriers)
                if (candidate.kind, candidate.atom) == identity
            )
            if len(positions) != 1:
                return None
            reordering.append(positions[0])
    return tuple(reordering)


def _transform_level2_descriptor(
    descriptor: _Level2Descriptor,
    source_carriers: Sequence[StereoCarrier],
    reordered_carriers: Sequence[StereoCarrier],
    *,
    source_anchor: StereoAnchor | None = None,
    reordered_anchor: StereoAnchor | None = None,
) -> _Level2AlgebraResult:
    """Transform a descriptor under an explicit carrier reordering."""

    family = descriptor.family
    count = _FAMILY_CARRIER_COUNTS[family]
    issues = []
    if (source_anchor is None) != (reordered_anchor is None):
        issues.append(
            _algebra_issue(
                StereoIssueCode.INVALID_ANCHOR,
                "source and reordered anchors must be supplied together",
            )
        )
    if source_anchor is not None and reordered_anchor is not None:
        issues.extend(_validate_stereo_anchor(source_anchor, family=family))
        issues.extend(_validate_stereo_anchor(reordered_anchor, family=family))
        if source_anchor.kind is not reordered_anchor.kind:
            same_identity = False
        elif source_anchor.kind is StereoAnchorKind.ATOM:
            same_identity = source_anchor.atom_indices == reordered_anchor.atom_indices
        elif source_anchor.kind is StereoAnchorKind.AXIS:
            same_identity = reordered_anchor.atom_indices in (
                source_anchor.atom_indices,
                source_anchor.atom_indices[::-1],
            )
        else:
            same_identity = set(source_anchor.atom_indices) == set(
                reordered_anchor.atom_indices
            )
        if not same_identity:
            issues.append(
                _algebra_issue(
                    StereoIssueCode.INVALID_ANCHOR,
                    "reordered anchor must identify the same graph object",
                )
            )
    carrier_validator = (
        _validate_gate2_double_bond_carriers
        if family is _Level2DescriptorFamily.AXIAL
        else lambda carriers: _validate_stereo_carriers(
            carriers,
            expected_count=count,
        )
    )
    issues.extend(carrier_validator(source_carriers))
    issues.extend(carrier_validator(reordered_carriers))
    if issues:
        return _Level2AlgebraResult(None, tuple(issues))

    anchor_reversed = (
        source_anchor is not None
        and reordered_anchor is not None
        and source_anchor.atom_indices == reordered_anchor.atom_indices[::-1]
    )
    reordering = (
        _endpoint_pair_reordering(
            source_carriers,
            reordered_carriers,
            endpoint_groups_reversed=anchor_reversed,
        )
        if family is _Level2DescriptorFamily.AXIAL
        else _carrier_reordering(source_carriers, reordered_carriers)
    )
    if reordering is None:
        return _Level2AlgebraResult(
            None,
            (
                _algebra_issue(
                    StereoIssueCode.INVALID_CARRIER,
                    "reordered carriers must contain the same identities",
                ),
            ),
        )

    if family in (
        _Level2DescriptorFamily.DOUBLE_BOND,
        _Level2DescriptorFamily.AXIAL,
    ) and not _is_endpoint_pair_reordering(reordering):
        return _Level2AlgebraResult(
            None,
            (
                _algebra_issue(
                    StereoIssueCode.INVALID_CARRIER,
                    f"{family.value} reordering must preserve or reverse "
                    "endpoint pairs",
                ),
            ),
        )

    if (
        family
        in (
            _Level2DescriptorFamily.DOUBLE_BOND,
            _Level2DescriptorFamily.AXIAL,
        )
        and source_anchor is not None
        and reordered_anchor is not None
    ):
        carrier_pairs_reversed = frozenset(reordering[:2]) == frozenset((2, 3))
        if anchor_reversed != carrier_pairs_reversed:
            return _Level2AlgebraResult(
                None,
                (
                    _algebra_issue(
                        StereoIssueCode.INVALID_CARRIER,
                        "ordered-anchor reversal and endpoint carrier-pair "
                        "order disagree",
                    ),
                ),
            )

    if family in (
        _Level2DescriptorFamily.TETRAHEDRAL,
        _Level2DescriptorFamily.AXIAL,
    ):
        parity = _PARITY_VALUES.index(descriptor.value)
        value = _PARITY_VALUES[parity ^ _permutation_parity(reordering)]
        return _Level2AlgebraResult(_Level2Descriptor(family, value), ())

    if family is _Level2DescriptorFamily.DOUBLE_BOND:
        relation = _ALKENE_VALUES.index(descriptor.value)
        value = _ALKENE_VALUES[relation ^ _permutation_parity(reordering)]
        return _Level2AlgebraResult(_Level2Descriptor(family, value), ())

    representatives = _COORDINATION_REPRESENTATIVES[family]
    representative = representatives[int(descriptor.value) - 1]
    inverse = [0] * len(reordering)
    for new_index, old_index in enumerate(reordering):
        inverse[old_index] = new_index
    transformed = tuple(inverse[item] for item in representative)
    rotations = _COORDINATION_ROTATIONS[family]
    canonical = min(
        tuple(transformed[position] for position in rotation)
        for rotation in rotations
    )
    value = representatives.index(canonical) + 1
    return _Level2AlgebraResult(_Level2Descriptor(family, value), ())


@dataclass(frozen=True)
class StereoDerivedDescriptor:
    """One derived name that is not part of stored stereo identity."""

    record_index: int
    stereo_type: StereoType
    label: str
    system: str
    is_absolute: bool


@dataclass(frozen=True)
class StereoAssignmentReport:
    """Assigned payload and facts that could not be resolved."""

    payload: StereoPayload
    unresolved: tuple[StereoValidationIssue, ...]
    derived: tuple[StereoDerivedDescriptor, ...] = ()


@dataclass(frozen=True)
class StereoAlignmentReport:
    """Stereo equivalence after applying an explicit atom mapping."""

    is_equivalent: bool
    aligned_payload: StereoPayload
    issues: tuple[StereoValidationIssue, ...]


def get_stereo_payload(mol: Molecule) -> StereoPayload:
    """Return the exact Level-1 payload stored by a molecule."""

    with mol._mutation_lock:
        return mol._stereo_payload


def set_stereo_payload(mol: Molecule, payload: StereoPayload) -> None:
    """Store Level-1 facts without chemical interpretation."""

    from .structure._commit import _commit_stereo
    from .structure._identity import _capture_source

    with mol._mutation_lock:
        _commit_stereo(mol, _capture_source(mol), payload)


_GATE6_ROW_FIELD = "_hotpot_stereo_row"
_GATE6_H_SLOT_FIELD = "_hotpot_stereo_h_slot"
_GATE6_ATOM_FIELDS = (
    "atomic_number",
    "isotope",
    "formal_charge",
    "radical_electrons",
    "is_aromatic",
)


@dataclass(frozen=True)
class _Gate6Graph:
    """One immutable augmented represented-graph snapshot for Gate 6."""

    target_spec: dict
    directed_dative_edges: frozenset[tuple[int, int]]
    real_atom_count: int
    virtual_hydrogens: tuple[tuple[int, ...], ...]


@dataclass(frozen=True)
class _Gate7CoordinationSite:
    """One base-valid declared coordination site behind the Gate-7 fence."""

    record_index: int
    descriptor: _Level2Descriptor
    center: int
    carrier_rows: tuple[int, ...]


@dataclass(frozen=True)
class _Gate7CoordinateCandidate:
    """One descriptor fit ranked by raw angular RMS in radians."""

    descriptor: _Level2Descriptor
    score_radians: float


class _Gate6Equivalence(Enum):
    """Tri-state result for one represented carrier-pair comparison."""

    EQUIVALENT = "equivalent"
    DISTINCT = "distinct"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class _Gate6SemanticContext:
    """Stereo facts retained while testing one record's carrier pair."""

    payload: StereoPayload
    descriptors: tuple[_Level2Descriptor, ...]
    targets: dict
    coordination_chelate_pairs: tuple[
        frozenset[tuple[int, int]] | None, ...
    ]
    opaque_records: tuple[StereoRecord, ...]
    pointwise_keys: frozenset[int]
    hydrogen_slot_exemptions: frozenset[int]


def _gate6_issue(
    message: str,
    *,
    atom_indices: tuple[int, ...] = (),
) -> StereoValidationIssue:
    return StereoValidationIssue(
        code=StereoIssueCode.UNSUPPORTED,
        message=message,
        atom_indices=atom_indices,
    )


def _gate6_semantics(
    degree: int,
    valence: float,
) -> dict:
    """Provide the scalar target shape required by ``NativeTarget``."""

    return {
        "degree": degree,
        "connectivity": degree,
        "valence": valence,
        "ring_count": 0,
        "ring_sizes": (),
    }


def _gate6_edge_is_cyclic(
    edge: tuple[int, int],
    adjacency: tuple[tuple[int, ...], ...],
) -> bool:
    """Return whether an edge lies on a cycle in the represented union graph."""

    start, finish = edge
    pending = [start]
    visited = {start}
    while pending:
        current = pending.pop()
        for neighbour in adjacency[current]:
            if frozenset((current, neighbour)) == frozenset(edge):
                continue
            if neighbour == finish:
                return True
            if neighbour not in visited:
                visited.add(neighbour)
                pending.append(neighbour)
    return False


def _gate6_projection(
    mol: Molecule,
) -> tuple[_Gate6Graph | None, tuple[StereoValidationIssue, ...]]:
    """Project all represented structure into one exact native-search graph.

    Active and both hidden bond classes participate. Implicit hydrogens become
    stable virtual protium nodes without a virtual-node identity label, so an
    ordinary explicit protium can remain graph-isomorphic to one.
    """

    atoms = tuple(mol.atoms)
    atom_count = len(atoms)
    payload = get_stereo_payload(mol)
    atom_facts: dict[int, AtomTransportRecord] = {}
    issues = []
    for fact in payload.atom_facts:
        row = fact.atom
        if (
            not isinstance(row, int)
            or isinstance(row, bool)
            or not 0 <= row < atom_count
        ):
            issues.append(
                _gate6_issue(
                    f"atom transport fact references out-of-range row {row!r}",
                )
            )
            continue
        if row in atom_facts:
            issues.append(
                _gate6_issue(
                    f"duplicate atom transport facts for row {row}",
                    atom_indices=(row,),
                )
            )
            continue
        atom_facts[row] = fact

    represented_bonds = tuple(
        mol.bonds + mol._hided_metal_bonds + mol._hided_covalent_bonds
    )
    atom_rows = {id(atom): row for row, atom in enumerate(atoms)}
    bonds_by_edge = {}
    raw_bonds = []
    for bond in represented_bonds:
        first = atom_rows.get(id(bond.atom1))
        second = atom_rows.get(id(bond.atom2))
        if first is None or second is None:
            issues.append(
                _gate6_issue(
                    "represented bond endpoint is outside the molecule atom table",
                )
            )
            continue
        if first == second:
            issues.append(
                _gate6_issue(
                    f"represented graph contains self-loop at row {first}",
                    atom_indices=(first,),
                )
            )
            continue
        edge = tuple(sorted((first, second)))
        if edge in bonds_by_edge:
            issues.append(
                _gate6_issue(
                    f"parallel represented bonds share edge {edge}",
                    atom_indices=edge,
                )
            )
            continue
        bonds_by_edge[edge] = bond
        order = float(bond.bond_order)
        if not isfinite(order):
            issues.append(
                _gate6_issue(
                    f"represented bond {edge} has non-finite order",
                    atom_indices=edge,
                )
            )
            continue
        kind = bond.bond_kind.value
        directed = None
        if kind == "dative":
            if bond.bond_direction == "atom1_to_atom2":
                directed = (first, second)
            elif bond.bond_direction == "atom2_to_atom1":
                directed = (second, first)
            else:
                issues.append(
                    _gate6_issue(
                        f"dative bond {edge} has unknown direction "
                        f"{bond.bond_direction!r}",
                        atom_indices=edge,
                    )
                )
                continue
        raw_bonds.append((edge, order, kind, directed))

    if issues:
        return None, tuple(issues)

    adjacency_lists = [[] for _ in atoms]
    for (first, second), _, _, _ in raw_bonds:
        adjacency_lists[first].append(second)
        adjacency_lists[second].append(first)
    adjacency = tuple(tuple(sorted(neighbours)) for neighbours in adjacency_lists)

    node_fields = []
    implicit_counts = []
    for row, atom in enumerate(atoms):
        fact = atom_facts.get(row)
        isotope = 0 if fact is None or fact.isotope is None else int(fact.isotope)
        if atom.atomic_number == 1 and isotope in (0, 1):
            isotope = 0
        radical = (
            0
            if fact is None or fact.radical_electrons is None
            else int(fact.radical_electrons)
        )
        node_fields.append(
            {
                "atomic_number": int(atom.atomic_number),
                "isotope": isotope,
                "formal_charge": int(atom.formal_charge),
                "radical_electrons": radical,
                "is_aromatic": bool(atom.is_aromatic),
                _GATE6_ROW_FIELD: row,
            }
        )
        implicit_count = int(atom.implicit_hydrogens)
        if implicit_count < 0:
            issues.append(
                _gate6_issue(
                    f"atom row {row} has negative implicit-hydrogen count",
                    atom_indices=(row,),
                )
            )
        implicit_counts.append(implicit_count)

    if issues:
        return None, tuple(issues)

    edges = []
    bond_records = []
    directed_dative_edges = set()
    for edge, order, kind, directed in sorted(raw_bonds):
        first, second = edge
        aromatic = (
            kind == "aromatic"
            or order == 1.5
            or (
                kind in ("single", "double")
                and atoms[first].is_aromatic
                and atoms[second].is_aromatic
                and _gate6_edge_is_cyclic(edge, adjacency)
            )
        )
        normalized_kind = "aromatic" if aromatic else kind
        normalized_order = 1.5 if aromatic else order
        edges.append(edge)
        bond_records.append(
            {
                "fields": {
                    "bond_order": normalized_order,
                    "is_aromatic": aromatic,
                },
                "bond_kind": normalized_kind,
                "is_aromatic": aromatic,
                "object": None,
            }
        )
        if directed is not None:
            directed_dative_edges.add(directed)

    next_key = atom_count
    virtual_hydrogens = [[] for _ in atoms]
    for row, hydrogen_count in enumerate(implicit_counts):
        for _ in range(hydrogen_count):
            hydrogen_key = next_key
            next_key += 1
            virtual_hydrogens[row].append(hydrogen_key)
            node_fields.append(
                {
                    "atomic_number": 1,
                    "isotope": 0,
                    "formal_charge": 0,
                    "radical_electrons": 0,
                    "is_aromatic": False,
                    _GATE6_ROW_FIELD: hydrogen_key,
                }
            )
            edges.append((row, hydrogen_key))
            bond_records.append(
                {
                    "fields": {
                        "bond_order": 1.0,
                        "is_aromatic": False,
                    },
                    "bond_kind": "single",
                    "is_aromatic": False,
                    "object": None,
                }
            )

    edge_records = sorted(zip(edges, bond_records), key=lambda item: item[0])
    edges = [edge for edge, _ in edge_records]
    bond_records = [record for _, record in edge_records]
    keys = tuple(range(next_key))
    neighbours = [[] for _ in keys]
    incident_bonds = [[] for _ in keys]
    for (first, second), bond_record in zip(edges, bond_records):
        neighbours[first].append(second)
        neighbours[second].append(first)
        incident_bonds[first].append((second, bond_record))
        incident_bonds[second].append((first, bond_record))
    hydrogen_groups = {}
    for key, fields in zip(keys, node_fields):
        if fields["atomic_number"] != 1 or len(neighbours[key]) != 1:
            continue
        parent = neighbours[key][0]
        _, bond_record = incident_bonds[key][0]
        direction = (
            "out"
            if (key, parent) in directed_dative_edges
            else (
                "in"
                if (parent, key) in directed_dative_edges
                else None
            )
        )
        signature = (
            *(fields[field] for field in _GATE6_ATOM_FIELDS),
            bond_record["bond_kind"],
            bond_record["fields"]["bond_order"],
            bond_record["fields"]["is_aromatic"],
            direction,
        )
        hydrogen_groups.setdefault((parent, signature), []).append(key)
    for hydrogen_keys in hydrogen_groups.values():
        for slot, key in enumerate(sorted(hydrogen_keys)):
            node_fields[key][_GATE6_H_SLOT_FIELD] = slot

    degrees = [0] * next_key
    valences = [0.0] * next_key
    for edge, bond_record in zip(edges, bond_records):
        order = float(bond_record["fields"]["bond_order"])
        for row in edge:
            degrees[row] += 1
            valences[row] += order
    atoms_spec = tuple(
        {
            "fields": fields,
            "full": _gate6_semantics(degrees[key], valences[key]),
            "ligand": _gate6_semantics(degrees[key], valences[key]),
            "object": None,
        }
        for key, fields in zip(keys, node_fields)
    )
    return (
        _Gate6Graph(
            target_spec={
                "keys": keys,
                "edges": tuple(edges),
                "atoms": atoms_spec,
                "bonds": tuple(bond_records),
            },
            directed_dative_edges=frozenset(directed_dative_edges),
            real_atom_count=atom_count,
            virtual_hydrogens=tuple(
                tuple(keys) for keys in virtual_hydrogens
            ),
        ),
        (),
    )


def _gate6_query_spec(
    graph: _Gate6Graph,
    pins: Mapping[int, int],
    *,
    hydrogen_slot_exemptions: Collection[int] | None = None,
) -> dict:
    """Build an exact whole-graph native query with optional symmetry pins.

    ``hydrogen_slot_exemptions=None`` omits every nonchemical H-slot
    constraint.  An explicit collection enables slots for all eligible
    hydrogens except the listed represented keys.
    """

    target = graph.target_spec
    nodes = []
    for key, atom in zip(target["keys"], target["atoms"]):
        leaves = [
            ("native_set", field, (atom["fields"][field],))
            for field in _GATE6_ATOM_FIELDS
        ]
        if key in pins:
            leaves.append(
                ("native_set", _GATE6_ROW_FIELD, (int(pins[key]),))
            )
        if (
            hydrogen_slot_exemptions is not None
            and key not in hydrogen_slot_exemptions
            and _GATE6_H_SLOT_FIELD in atom["fields"]
        ):
            leaves.append(
                (
                    "native_set",
                    _GATE6_H_SLOT_FIELD,
                    (atom["fields"][_GATE6_H_SLOT_FIELD],),
                )
            )
        nodes.append(
            {
                "preflight": (),
                "leaves": tuple(leaves),
            }
        )

    bonds = tuple(
        {
            "preflight": (),
            "leaves": tuple(
                ("native_set", field, (value,))
                for field, value in bond["fields"].items()
            ),
            "exclude_aromatic": not bond["is_aromatic"],
            "bond_kind": bond["bond_kind"],
        }
        for bond in target["bonds"]
    )
    return {
        "keys": target["keys"],
        "edges": target["edges"],
        "nodes": tuple(nodes),
        "bonds": bonds,
    }


def _gate6_carrier_keys(
    record: StereoRecord,
    descriptor: _Level2Descriptor,
    graph: _Gate6Graph,
) -> tuple[int, ...] | None:
    """Resolve one record's carriers into augmented represented-graph keys."""

    if descriptor.family in _COORDINATION_FAMILIES:
        if any(
            carrier.kind is not StereoCarrierKind.ATOM
            for carrier in record.carriers
        ):
            return None
        return tuple(carrier.atom for carrier in record.carriers)
    if descriptor.family is _Level2DescriptorFamily.TETRAHEDRAL:
        endpoints = (record.anchor.atom_indices[0],) * len(record.carriers)
    else:
        endpoints = (
            (record.anchor.atom_indices[0],) * 2
            + (record.anchor.atom_indices[-1],) * 2
        )

    keys = []
    for carrier, endpoint in zip(record.carriers, endpoints):
        if carrier.kind is StereoCarrierKind.ATOM:
            keys.append(carrier.atom)
            continue
        if carrier.kind is not StereoCarrierKind.IMPLICIT_H:
            return None
        virtual = graph.virtual_hydrogens[endpoint]
        if len(virtual) != 1:
            return None
        keys.append(virtual[0])
    return tuple(keys)


def _gate6_materialize_record(
    record: StereoRecord,
    descriptor: _Level2Descriptor,
    graph: _Gate6Graph,
) -> StereoRecord | None:
    """Replace supported implicit-H carriers with represented atom tokens."""

    carrier_keys = _gate6_carrier_keys(record, descriptor, graph)
    if carrier_keys is None:
        return None
    return replace(
        record,
        carriers=tuple(
            StereoCarrier(StereoCarrierKind.ATOM, key)
            for key in carrier_keys
        ),
    )


def _gate6_semantic_context(
    payload: StereoPayload,
    graph: _Gate6Graph,
    *,
    excluded_record_index: int,
    valid_record_indices: Collection[int],
    unsupported_record_indices: Collection[int],
) -> _Gate6SemanticContext:
    """Prepare all descendant stereo semantics except the record under test."""

    valid_record_indices = set(valid_record_indices)
    unsupported_record_indices = set(unsupported_record_indices)
    records = []
    descriptors = []
    opaque_records = []
    for record_index, record in enumerate(payload.records):
        if record_index == excluded_record_index:
            continue
        parsed = _parse_level2_descriptor(record.descriptor)
        descriptor = parsed.descriptor
        if (
            record_index in valid_record_indices
            and descriptor is not None
            and descriptor.family in (
                _Level2DescriptorFamily.TETRAHEDRAL,
                _Level2DescriptorFamily.DOUBLE_BOND,
                _Level2DescriptorFamily.AXIAL,
                *_COORDINATION_FAMILIES,
            )
        ):
            materialized = _gate6_materialize_record(
                record,
                descriptor,
                graph,
            )
            if materialized is not None:
                records.append(materialized)
                descriptors.append(descriptor)
                continue
            opaque_records.append(record)
            continue
        if (
            record_index in unsupported_record_indices
            and record.descriptor.startswith(
                f"{_LEVEL2_DESCRIPTOR_PREFIX}/"
            )
        ):
            opaque_records.append(record)

    reduced_payload = replace(payload, records=tuple(records))
    targets = {}
    for record_index, (record, descriptor) in enumerate(
        zip(reduced_payload.records, descriptors)
    ):
        identity = _alignment_record_identity(descriptor.family, record.anchor)
        targets.setdefault(identity, []).append(
            (record_index, record, descriptor)
        )
    represented_adjacency = _gate7_represented_adjacency(graph)
    coordination_chelate_pairs = tuple(
        _gate7_chelate_pairs(
            _gate7_site_from_record(record_index, record, descriptor),
            represented_adjacency,
        )
        if descriptor.family in _COORDINATION_FAMILIES
        else None
        for record_index, (record, descriptor) in enumerate(
            zip(reduced_payload.records, descriptors)
        )
    )
    semantic_records = (
        payload.records[excluded_record_index],
        *reduced_payload.records,
        *opaque_records,
    )
    semantic_keys = {
        atom
        for record in semantic_records
        for atom in _record_atom_indices(record)
        if 0 <= atom < len(graph.target_spec["keys"])
    }
    adjacency = {
        key: set() for key in graph.target_spec["keys"]
    }
    for first, second in graph.target_spec["edges"]:
        adjacency[first].add(second)
        adjacency[second].add(first)
    pointwise_keys = set()
    unseen = set(adjacency)
    while unseen:
        seed = min(unseen)
        component = set()
        pending = [seed]
        while pending:
            key = pending.pop()
            if key in component:
                continue
            component.add(key)
            pending.extend(adjacency[key] - component)
        unseen.difference_update(component)
        if component.isdisjoint(semantic_keys):
            pointwise_keys.update(component)
    hydrogen_slot_exemptions = {
        carrier.atom
        for record in reduced_payload.records
        for carrier in record.carriers
        if (
            carrier.kind is StereoCarrierKind.ATOM
            and graph.target_spec["atoms"][carrier.atom]["fields"][
                "atomic_number"
            ]
            == 1
        )
    }
    for record in opaque_records:
        hydrogen_slot_exemptions.update(
            atom
            for atom in _record_atom_indices(record)
            if (
                0 <= atom < len(graph.target_spec["keys"])
                and graph.target_spec["atoms"][atom]["fields"][
                    "atomic_number"
                ]
                == 1
            )
        )
        if any(
            carrier.kind is not StereoCarrierKind.ATOM
            for carrier in record.carriers
        ):
            hydrogen_slot_exemptions.update(
                hydrogen
                for atom in record.anchor.atom_indices
                if (
                    isinstance(atom, int)
                    and not isinstance(atom, bool)
                    and 0 <= atom < graph.real_atom_count
                )
                for hydrogen in graph.virtual_hydrogens[atom]
            )
    return _Gate6SemanticContext(
        payload=reduced_payload,
        descriptors=tuple(descriptors),
        targets=targets,
        coordination_chelate_pairs=coordination_chelate_pairs,
        opaque_records=tuple(opaque_records),
        pointwise_keys=frozenset(pointwise_keys),
        hydrogen_slot_exemptions=frozenset(hydrogen_slot_exemptions),
    )


def _gate6_dative_direction_is_preserved(
    graph: _Gate6Graph,
    mapping: Mapping[int, int],
) -> bool:
    """Compare the complete directed-dative edge set under one mapping."""

    return frozenset(
        (mapping[begin], mapping[end])
        for begin, end in graph.directed_dative_edges
    ) == graph.directed_dative_edges


def _gate6_opaque_record_is_fixed(
    record: StereoRecord,
    graph: _Gate6Graph,
    mapping: Mapping[int, int],
) -> bool:
    """Accept an uninterpreted Hotpot record only when it remains pointwise fixed."""

    if any(
        mapping.get(atom) != atom
        for atom in _record_atom_indices(record)
    ):
        return False
    if any(
        carrier.kind is not StereoCarrierKind.ATOM
        for carrier in record.carriers
    ):
        return all(
            mapping[hydrogen] == hydrogen
            for atom in record.anchor.atom_indices
            if (
                isinstance(atom, int)
                and not isinstance(atom, bool)
                and 0 <= atom < graph.real_atom_count
            )
            for hydrogen in graph.virtual_hydrogens[atom]
        ) and all(
            isinstance(atom, int)
            and not isinstance(atom, bool)
            and 0 <= atom < graph.real_atom_count
            for atom in record.anchor.atom_indices
        )
    return True


def _transform_supported_record(
    descriptor: _Level2Descriptor,
    mapped: StereoRecord,
    target: StereoRecord,
) -> _Level2AlgebraResult:
    """Apply the frozen family-specific descriptor transformation."""

    if descriptor.family in _ATOM_GEOMETRY_FAMILIES:
        return _transform_level2_descriptor(
            descriptor,
            mapped.carriers,
            target.carriers,
            source_anchor=mapped.anchor,
            reordered_anchor=target.anchor,
        )
    if descriptor.family is _Level2DescriptorFamily.DOUBLE_BOND:
        return _transform_gate3_double_bond(descriptor, mapped, target)
    return _transform_gate5_axial(descriptor, mapped, target)


def _gate6_mapping_preserves_stereo(
    graph: _Gate6Graph,
    context: _Gate6SemanticContext,
    mapping: Mapping[int, int],
) -> _Gate6Equivalence:
    """Classify one topology automorphism after exact stereo postfilters."""

    if not _gate6_dative_direction_is_preserved(graph, mapping):
        return _Gate6Equivalence.DISTINCT

    matches = []
    matched_target_indices = set()
    for reference_index, (record, descriptor) in enumerate(
        zip(context.payload.records, context.descriptors)
    ):
        mapped = _mapped_alignment_record(record, mapping)
        identity = _alignment_record_identity(descriptor.family, mapped.anchor)
        targets = context.targets.get(identity, ())
        if len(targets) != 1:
            return _Gate6Equivalence.DISTINCT
        target_index, target, _ = targets[0]
        transformed = _transform_supported_record(descriptor, mapped, target)
        if transformed.descriptor is None or transformed.issues:
            return _Gate6Equivalence.DISTINCT
        if descriptor.family in _COORDINATION_FAMILIES:
            source_pairs = context.coordination_chelate_pairs[reference_index]
            mapped_pairs = _gate7_mapped_chelate_pairs(source_pairs, mapping)
            if mapped_pairs != context.coordination_chelate_pairs[target_index]:
                return _Gate6Equivalence.DISTINCT
        matched_target_indices.add(target_index)
        matches.append(
            _AlignmentMatch(
                reference_index=reference_index,
                target_index=target_index,
                mapped=mapped,
                target=target,
                transformed=transformed.descriptor,
            )
        )

    if matched_target_indices != set(range(len(context.payload.records))):
        return _Gate6Equivalence.DISTINCT
    if _alignment_group_issues(context.payload, context.payload, matches):
        return _Gate6Equivalence.DISTINCT
    if any(
        not _gate6_opaque_record_is_fixed(record, graph, mapping)
        for record in context.opaque_records
    ):
        return _Gate6Equivalence.UNKNOWN
    return _Gate6Equivalence.EQUIVALENT


def _gate6_carrier_pair_equivalence(
    native_target,
    graph: _Gate6Graph,
    context: _Gate6SemanticContext,
    record: StereoRecord,
    carrier_keys: tuple[int, ...],
    pair: tuple[int, int],
) -> _Gate6Equivalence:
    """Test one carrier pair by pinned whole-graph RI automorphisms."""

    pins = {key: key for key in context.pointwise_keys}
    pins.update((atom, atom) for atom in record.anchor.atom_indices)
    pins[carrier_keys[pair[0]]] = carrier_keys[pair[1]]
    saw_unknown = False
    for mapping in native_target.iter_mappings(
        _gate6_query_spec(
            graph,
            pins,
            hydrogen_slot_exemptions=(
                context.hydrogen_slot_exemptions
                | {
                    carrier_keys[pair[0]],
                    carrier_keys[pair[1]],
                }
            ),
        )
    ):
        result = _gate6_mapping_preserves_stereo(graph, context, mapping)
        if result is _Gate6Equivalence.EQUIVALENT:
            return result
        if result is _Gate6Equivalence.UNKNOWN:
            saw_unknown = True
    return (
        _Gate6Equivalence.UNKNOWN
        if saw_unknown
        else _Gate6Equivalence.DISTINCT
    )


def _gate6_stereogenicity_issues(
    mol: Molecule,
    graph_report: StereoValidationReport,
    *,
    projection_result: tuple[
        _Gate6Graph | None, tuple[StereoValidationIssue, ...]
    ]
    | None = None,
) -> tuple[StereoValidationIssue, ...]:
    """Prove supported carrier distinction on the exact represented graph."""

    payload = get_stereo_payload(mol)
    invalid_record_indices = {
        issue.record_index
        for issue in graph_report.issues
        if issue.record_index is not None
    }
    unsupported_record_indices = {
        issue.record_index
        for issue in graph_report.issues
        if (
            issue.record_index is not None
            and issue.code is StereoIssueCode.UNSUPPORTED
        )
    }
    candidates = []
    valid_record_indices = set()
    for record_index, record in enumerate(payload.records):
        parsed = _parse_level2_descriptor(record.descriptor)
        descriptor = parsed.descriptor
        if (
            record_index not in invalid_record_indices
            and descriptor is not None
            and not parsed.issues
            and descriptor.family in (
                _Level2DescriptorFamily.TETRAHEDRAL,
                _Level2DescriptorFamily.DOUBLE_BOND,
                _Level2DescriptorFamily.AXIAL,
                *_COORDINATION_FAMILIES,
            )
        ):
            valid_record_indices.add(record_index)
            if descriptor.family not in _COORDINATION_FAMILIES:
                candidates.append((record_index, record, descriptor))
    if not candidates:
        return ()

    graph, projection_issues = (
        _gate6_projection(mol)
        if projection_result is None
        else projection_result
    )
    if graph is None:
        return projection_issues if projection_result is None else ()

    from .search.native import _NativeTarget

    native_target = _NativeTarget(graph.target_spec)
    issues = []
    for record_index, record, descriptor in candidates:
        carrier_keys = _gate6_carrier_keys(record, descriptor, graph)
        if carrier_keys is None:
            issues.append(
                _gate2_issue(
                    StereoIssueCode.UNSUPPORTED,
                    "Gate-6 represented carrier identity is unresolved",
                    record_index=record_index,
                    atom_indices=_record_atom_indices(record),
                )
            )
            continue

        context = _gate6_semantic_context(
            payload,
            graph,
            excluded_record_index=record_index,
            valid_record_indices=valid_record_indices,
            unsupported_record_indices=unsupported_record_indices,
        )
        pairs = (
            tuple(combinations(range(4), 2))
            if descriptor.family is _Level2DescriptorFamily.TETRAHEDRAL
            else ((0, 1), (2, 3))
        )
        saw_unknown = False
        equivalent_pair = None
        for pair in pairs:
            result = _gate6_carrier_pair_equivalence(
                native_target,
                graph,
                context,
                record,
                carrier_keys,
                pair,
            )
            if result is _Gate6Equivalence.EQUIVALENT:
                equivalent_pair = pair
                break
            if result is _Gate6Equivalence.UNKNOWN:
                saw_unknown = True

        if equivalent_pair is not None:
            issues.append(
                _gate2_issue(
                    StereoIssueCode.NON_STEREOGENIC,
                    "represented graph makes carrier positions "
                    f"{equivalent_pair[0]} and {equivalent_pair[1]} equivalent",
                    record_index=record_index,
                    atom_indices=_record_atom_indices(record),
                )
            )
        elif saw_unknown:
            issues.append(
                _gate2_issue(
                    StereoIssueCode.UNSUPPORTED,
                    "represented-graph carrier equivalence depends on moved "
                    "unsupported stereochemistry",
                    record_index=record_index,
                    atom_indices=_record_atom_indices(record),
                )
            )
    return tuple(issues)


def _record_atom_indices(record: StereoRecord) -> tuple[int, ...]:
    """Return record atom rows once, in their stored reference order."""

    rows = []
    for atom in record.anchor.atom_indices:
        if isinstance(atom, int) and not isinstance(atom, bool) and atom not in rows:
            rows.append(atom)
    for carrier in record.carriers:
        if (
            carrier.kind is StereoCarrierKind.ATOM
            and isinstance(carrier.atom, int)
            and not isinstance(carrier.atom, bool)
            and carrier.atom not in rows
        ):
            rows.append(carrier.atom)
    return tuple(rows)


def _record_issues(
    issues: Sequence[StereoValidationIssue],
    *,
    record_index: int,
    atom_indices: tuple[int, ...],
) -> tuple[StereoValidationIssue, ...]:
    """Attach stable record context to chemistry-independent diagnostics."""

    return tuple(
        replace(
            issue,
            record_index=record_index,
            atom_indices=issue.atom_indices or atom_indices,
        )
        for issue in issues
    )


def _gate2_issue(
    code: StereoIssueCode,
    message: str,
    *,
    record_index: int,
    atom_indices: tuple[int, ...] = (),
) -> StereoValidationIssue:
    return StereoValidationIssue(
        code=code,
        message=message,
        record_index=record_index,
        atom_indices=atom_indices,
    )


def _validate_gate2_double_bond_carriers(
    carriers: Sequence[StereoCarrier],
) -> tuple[StereoValidationIssue, ...]:
    """Validate DB carrier shape with virtual-H identity scoped per endpoint."""

    issues = []
    if len(carriers) != 4:
        issues.append(
            _algebra_issue(
                StereoIssueCode.INVALID_CARRIER,
                f"expected 4 ordered carriers, got {len(carriers)}",
            )
        )

    for position, carrier in enumerate(carriers):
        if carrier.kind is StereoCarrierKind.ATOM:
            if (
                not isinstance(carrier.atom, int)
                or isinstance(carrier.atom, bool)
                or carrier.atom < 0
            ):
                issues.append(
                    _algebra_issue(
                        StereoIssueCode.INVALID_CARRIER,
                        f"atom carrier at position {position} requires a "
                        "non-negative atom row",
                    )
                )
        elif carrier.atom is not None:
            issues.append(
                _algebra_issue(
                    StereoIssueCode.INVALID_CARRIER,
                    f"non-atom carrier at position {position} cannot carry an atom row",
                )
            )

    for endpoint, start in enumerate((0, 2)):
        identities = [
            (carrier.kind, carrier.atom)
            for carrier in carriers[start : start + 2]
        ]
        if len(set(identities)) != len(identities):
            issues.append(
                _algebra_issue(
                    StereoIssueCode.INVALID_CARRIER,
                    f"duplicate carrier identity in endpoint group {endpoint}",
                )
            )
    return tuple(issues)


def _validate_gate5_axial_carriers(
    carriers: Sequence[StereoCarrier],
) -> tuple[StereoValidationIssue, ...]:
    """Validate endpoint-scoped AX carrier identity and virtual hydrogens."""

    issues = []
    if len(carriers) != 4:
        issues.append(
            _algebra_issue(
                StereoIssueCode.INVALID_CARRIER,
                f"expected 4 ordered carriers, got {len(carriers)}",
            )
        )
    for position, carrier in enumerate(carriers):
        if carrier.kind is StereoCarrierKind.ATOM:
            if (
                not isinstance(carrier.atom, int)
                or isinstance(carrier.atom, bool)
                or carrier.atom < 0
            ):
                issues.append(
                    _algebra_issue(
                        StereoIssueCode.INVALID_CARRIER,
                        f"atom carrier at position {position} requires a "
                        "non-negative atom row",
                    )
                )
        elif carrier.kind is not StereoCarrierKind.IMPLICIT_H:
            issues.append(
                _algebra_issue(
                    StereoIssueCode.INVALID_CARRIER,
                    "AX permits only atom and implicit-H carriers",
                )
            )
        if carrier.kind is not StereoCarrierKind.ATOM and carrier.atom is not None:
            issues.append(
                _algebra_issue(
                    StereoIssueCode.INVALID_CARRIER,
                    f"non-atom carrier at position {position} cannot carry an atom row",
                )
            )

    for endpoint, start in enumerate((0, 2)):
        group = carriers[start : start + 2]
        identities = tuple((carrier.kind, carrier.atom) for carrier in group)
        if len(set(identities)) != len(identities):
            issues.append(
                _algebra_issue(
                    StereoIssueCode.INVALID_CARRIER,
                    f"duplicate carrier identity in AX endpoint group {endpoint}",
                )
            )
        if sum(
            carrier.kind is StereoCarrierKind.IMPLICIT_H for carrier in group
        ) > 1:
            issues.append(
                _algebra_issue(
                    StereoIssueCode.INVALID_CARRIER,
                    f"AX endpoint group {endpoint} permits at most one "
                    "implicit-H carrier",
                )
            )
    return tuple(issues)


def _molecule_stereo_topology(
    mol: Molecule,
) -> tuple[int, tuple[frozenset[int], ...], dict[frozenset[int], float]]:
    """Return atom count, adjacency, and structural bond orders by atom row."""

    atom_count = len(mol.atoms)
    adjacency = [set() for _ in range(atom_count)]
    bond_orders = {}
    atom_rows = {id(atom): row for row, atom in enumerate(mol._atoms)}
    for bond in mol._bonds + mol._hided_bonds:
        begin = atom_rows[id(bond.atom1)]
        end = atom_rows[id(bond.atom2)]
        adjacency[begin].add(end)
        adjacency[end].add(begin)
        bond_orders[frozenset((begin, end))] = float(bond.bond_order)
    return (
        atom_count,
        tuple(frozenset(neighbours) for neighbours in adjacency),
        bond_orders,
    )


def _molecule_stereo_bond_kinds(mol: Molecule) -> dict[frozenset[int], str]:
    """Return semantic bond kinds without invoking toolkit perception."""

    atom_rows = {id(atom): row for row, atom in enumerate(mol._atoms)}
    return {
        frozenset((atom_rows[id(bond.atom1)], atom_rows[id(bond.atom2)])):
        bond.bond_kind.value
        for bond in mol._bonds + mol._hided_bonds
    }


def _validate_gate2_atom_ranges(
    record: StereoRecord,
    *,
    record_index: int,
    atom_count: int,
) -> tuple[StereoValidationIssue, ...]:
    """Validate well-formed anchor and carrier rows against the molecule."""

    issues = []
    for atom in record.anchor.atom_indices:
        if (
            isinstance(atom, int)
            and not isinstance(atom, bool)
            and atom >= 0
            and atom >= atom_count
        ):
            issues.append(
                _gate2_issue(
                    StereoIssueCode.INVALID_ANCHOR,
                    f"anchor atom row {atom} is outside molecule range "
                    f"0..{atom_count - 1}",
                    record_index=record_index,
                    atom_indices=(atom,),
                )
            )
    for carrier in record.carriers:
        atom = carrier.atom
        if (
            carrier.kind is StereoCarrierKind.ATOM
            and isinstance(atom, int)
            and not isinstance(atom, bool)
            and atom >= 0
            and atom >= atom_count
        ):
            issues.append(
                _gate2_issue(
                    StereoIssueCode.INVALID_CARRIER,
                    f"carrier atom row {atom} is outside molecule range "
                    f"0..{atom_count - 1}",
                    record_index=record_index,
                    atom_indices=(atom,),
                )
            )
    return tuple(issues)


def _validate_gate2_tetrahedral_graph(
    record: StereoRecord,
    *,
    record_index: int,
    atom_count: int,
    adjacency: Sequence[frozenset[int]],
) -> tuple[StereoValidationIssue, ...]:
    issues = []
    if (
        record.anchor.kind is not StereoAnchorKind.ATOM
        or len(record.anchor.atom_indices) != 1
    ):
        return ()

    center = record.anchor.atom_indices[0]
    if not 0 <= center < atom_count:
        return ()

    atom_carriers = set()
    virtual_hydrogens = 0
    for position, carrier in enumerate(record.carriers):
        if carrier.kind is StereoCarrierKind.ATOM:
            atom = carrier.atom
            if not isinstance(atom, int) or isinstance(atom, bool) or atom < 0:
                continue
            if atom >= atom_count:
                continue
            elif atom not in adjacency[center]:
                issues.append(
                    _gate2_issue(
                        StereoIssueCode.INVALID_CARRIER,
                        f"carrier atom row {atom} is not adjacent to "
                        f"TH center {center}",
                        record_index=record_index,
                        atom_indices=(center, atom),
                    )
                )
            atom_carriers.add(atom)
        elif carrier.kind is StereoCarrierKind.IMPLICIT_H:
            virtual_hydrogens += 1
        else:
            issues.append(
                _gate2_issue(
                    StereoIssueCode.INVALID_CARRIER,
                    "TH permits only atom and implicit-H carriers",
                    record_index=record_index,
                    atom_indices=(center,),
                )
            )

    if virtual_hydrogens > 1:
        issues.append(
            _gate2_issue(
                StereoIssueCode.INVALID_CARRIER,
                "TH permits at most one implicit-H carrier",
                record_index=record_index,
                atom_indices=(center,),
            )
        )
    if atom_carriers != set(adjacency[center]):
        issues.append(
            _gate2_issue(
                StereoIssueCode.INVALID_CARRIER,
                "TH atom carriers must exactly cover the center's graph neighbours",
                record_index=record_index,
                atom_indices=(center, *sorted(atom_carriers | set(adjacency[center]))),
            )
        )
    if len(adjacency[center]) + virtual_hydrogens != 4:
        issues.append(
            _gate2_issue(
                StereoIssueCode.INVALID_CARRIER,
                "TH requires exactly four graph or implicit-H carriers",
                record_index=record_index,
                atom_indices=(center,),
            )
        )
    return tuple(issues)


def _validate_gate2_double_bond_graph(
    record: StereoRecord,
    *,
    record_index: int,
    atom_count: int,
    adjacency: Sequence[frozenset[int]],
    bond_orders: Mapping[frozenset[int], float],
) -> tuple[StereoValidationIssue, ...]:
    issues = []
    if (
        record.anchor.kind is not StereoAnchorKind.BOND
        or len(record.anchor.atom_indices) != 2
    ):
        return ()

    begin, end = record.anchor.atom_indices
    invalid_endpoints = tuple(
        atom for atom in (begin, end) if not 0 <= atom < atom_count
    )
    if invalid_endpoints:
        return ()

    bond_key = frozenset((begin, end))
    if bond_key not in bond_orders:
        issues.append(
            _gate2_issue(
                StereoIssueCode.INVALID_ANCHOR,
                "DB anchor endpoints are not bonded in the molecule",
                record_index=record_index,
                atom_indices=(begin, end),
            )
        )
    elif bond_orders[bond_key] != 2.0:
        issues.append(
            _gate2_issue(
                StereoIssueCode.INVALID_ANCHOR,
                "DB anchor must identify a double bond",
                record_index=record_index,
                atom_indices=(begin, end),
            )
        )

    for endpoint_index, (endpoint, other) in enumerate(((begin, end), (end, begin))):
        group = record.carriers[2 * endpoint_index : 2 * endpoint_index + 2]
        atom_carriers = set()
        virtual_hydrogens = 0
        for carrier in group:
            if carrier.kind is StereoCarrierKind.ATOM:
                atom = carrier.atom
                if not isinstance(atom, int) or isinstance(atom, bool) or atom < 0:
                    continue
                if atom >= atom_count:
                    continue
                elif atom == other or atom not in adjacency[endpoint]:
                    issues.append(
                        _gate2_issue(
                            StereoIssueCode.INVALID_CARRIER,
                            f"carrier atom row {atom} is not a substituent "
                            f"of DB endpoint {endpoint}",
                            record_index=record_index,
                            atom_indices=(endpoint, atom),
                        )
                    )
                atom_carriers.add(atom)
            elif carrier.kind is StereoCarrierKind.IMPLICIT_H:
                virtual_hydrogens += 1
            else:
                issues.append(
                    _gate2_issue(
                        StereoIssueCode.INVALID_CARRIER,
                        "DB permits only atom and implicit-H carriers",
                        record_index=record_index,
                        atom_indices=(endpoint,),
                    )
                )

        neighbours = set(adjacency[endpoint]) - {other}
        if atom_carriers != neighbours:
            issues.append(
                _gate2_issue(
                    StereoIssueCode.INVALID_CARRIER,
                    "DB endpoint atom carriers must exactly cover its graph "
                    "substituents",
                    record_index=record_index,
                    atom_indices=(endpoint, *sorted(atom_carriers | neighbours)),
                )
            )
        if virtual_hydrogens > 1:
            issues.append(
                _gate2_issue(
                    StereoIssueCode.INVALID_CARRIER,
                    "each DB endpoint permits at most one implicit-H carrier",
                    record_index=record_index,
                    atom_indices=(endpoint,),
                )
            )
        if len(neighbours) + virtual_hydrogens != 2:
            issues.append(
                _gate2_issue(
                    StereoIssueCode.INVALID_CARRIER,
                    "each DB endpoint requires exactly two graph or "
                    "implicit-H carriers",
                    record_index=record_index,
                    atom_indices=(endpoint,),
                )
            )
    return tuple(issues)


def _validate_gate5_axial_graph(
    record: StereoRecord,
    *,
    record_index: int,
    atom_count: int,
    adjacency: Sequence[frozenset[int]],
    bond_orders: Mapping[frozenset[int], float],
    bond_kinds: Mapping[frozenset[int], str],
) -> tuple[StereoValidationIssue, ...]:
    """Validate explicit allene/cumulene and formal atrop AX graph identity."""

    anchor = record.anchor
    path = anchor.atom_indices
    if any(not 0 <= atom < atom_count for atom in path):
        return ()

    issues = []
    if anchor.kind is StereoAnchorKind.AXIS:
        path_edges = tuple(
            frozenset((begin, end)) for begin, end in zip(path, path[1:])
        )
        for edge in path_edges:
            if edge not in bond_orders:
                issues.append(
                    _gate2_issue(
                        StereoIssueCode.INVALID_ANCHOR,
                        "AX axis path contains atoms that are not bonded",
                        record_index=record_index,
                        atom_indices=path,
                    )
                )
            elif bond_orders[edge] != 2.0 or bond_kinds[edge] != "double":
                issues.append(
                    _gate2_issue(
                        StereoIssueCode.INVALID_ANCHOR,
                        "AX axis path edges must be non-aromatic double bonds",
                        record_index=record_index,
                        atom_indices=path,
                    )
                )

        for position, atom in enumerate(path[1:-1], start=1):
            off_axis = set(adjacency[atom]) - {path[position - 1], path[position + 1]}
            if off_axis:
                issues.append(
                    _gate2_issue(
                        StereoIssueCode.INVALID_ANCHOR,
                        "AX axis path internal atoms cannot have off-axis neighbours",
                        record_index=record_index,
                        atom_indices=(atom, *sorted(off_axis)),
                    )
                )

        for endpoint, next_atom in ((path[0], path[1]), (path[-1], path[-2])):
            extends_axis = any(
                bond_orders.get(frozenset((endpoint, neighbour))) == 2.0
                and bond_kinds.get(frozenset((endpoint, neighbour))) == "double"
                for neighbour in adjacency[endpoint] - {next_atom}
            )
            if extends_axis:
                issues.append(
                    _gate2_issue(
                        StereoIssueCode.INVALID_ANCHOR,
                        "AX axis path must be maximal across non-aromatic double bonds",
                        record_index=record_index,
                        atom_indices=path,
                    )
                )
    else:
        begin, end = path
        edge = frozenset((begin, end))
        if edge not in bond_orders:
            issues.append(
                _gate2_issue(
                    StereoIssueCode.INVALID_ANCHOR,
                    "formal atrop AX anchor endpoints are not bonded",
                    record_index=record_index,
                    atom_indices=path,
                )
            )
        elif bond_orders[edge] != 1.0 or bond_kinds[edge] != "single":
            issues.append(
                _gate2_issue(
                    StereoIssueCode.INVALID_ANCHOR,
                    "formal atrop AX anchor must identify a non-aromatic, "
                    "non-dative single bond",
                    record_index=record_index,
                    atom_indices=path,
                )
            )

    begin, end = path[0], path[-1]
    adjacent_axis_atoms = path[1], path[-2]
    for endpoint_index, (endpoint, axis_neighbour) in enumerate(
        zip((begin, end), adjacent_axis_atoms)
    ):
        group = record.carriers[2 * endpoint_index : 2 * endpoint_index + 2]
        atom_carriers = set()
        virtual_hydrogens = 0
        for carrier in group:
            if carrier.kind is StereoCarrierKind.ATOM:
                atom = carrier.atom
                if (
                    not isinstance(atom, int)
                    or isinstance(atom, bool)
                    or not 0 <= atom < atom_count
                ):
                    continue
                if atom == axis_neighbour or atom not in adjacency[endpoint]:
                    issues.append(
                        _gate2_issue(
                            StereoIssueCode.INVALID_CARRIER,
                            f"carrier atom row {atom} is not an off-axis neighbour "
                            f"of AX endpoint {endpoint}",
                            record_index=record_index,
                            atom_indices=(endpoint, atom),
                        )
                    )
                atom_carriers.add(atom)
            elif carrier.kind is StereoCarrierKind.IMPLICIT_H:
                virtual_hydrogens += 1

        neighbours = set(adjacency[endpoint]) - {axis_neighbour}
        if atom_carriers != neighbours:
            issues.append(
                _gate2_issue(
                    StereoIssueCode.INVALID_CARRIER,
                    "AX endpoint atom carriers must exactly cover its off-axis "
                    "graph neighbours",
                    record_index=record_index,
                    atom_indices=(endpoint, *sorted(atom_carriers | neighbours)),
                )
            )
        if virtual_hydrogens > 1:
            issues.append(
                _gate2_issue(
                    StereoIssueCode.INVALID_CARRIER,
                    "each AX endpoint permits at most one implicit-H carrier",
                    record_index=record_index,
                    atom_indices=(endpoint,),
                )
            )
        if len(neighbours) + virtual_hydrogens != 2:
            issues.append(
                _gate2_issue(
                    StereoIssueCode.INVALID_CARRIER,
                    "each AX endpoint requires exactly two graph or implicit-H "
                    "carriers",
                    record_index=record_index,
                    atom_indices=(endpoint,),
                )
            )
    return tuple(issues)


def _gate7_represented_adjacency(
    graph: _Gate6Graph,
) -> tuple[frozenset[int], ...]:
    """Return the complete undirected Gate-6 current-state adjacency."""

    adjacency = [set() for _ in graph.target_spec["keys"]]
    for first, second in graph.target_spec["edges"]:
        adjacency[first].add(second)
        adjacency[second].add(first)
    return tuple(frozenset(neighbours) for neighbours in adjacency)


def _gate7_same_center_issues(
    payload: StereoPayload,
    descriptors: Mapping[int, _Level2Descriptor],
    *,
    atom_count: int,
) -> tuple[StereoValidationIssue, ...]:
    """Reject every parsed atom-geometry declaration sharing one center."""

    records_by_center: dict[int, list[int]] = {}
    for record_index, record in enumerate(payload.records):
        descriptor = descriptors.get(record_index)
        if descriptor is None or descriptor.family not in _ATOM_GEOMETRY_FAMILIES:
            continue
        anchor = record.anchor
        if anchor.kind is not StereoAnchorKind.ATOM or len(anchor.atom_indices) != 1:
            continue
        center = anchor.atom_indices[0]
        if (
            not isinstance(center, int)
            or isinstance(center, bool)
            or not 0 <= center < atom_count
        ):
            continue
        records_by_center.setdefault(center, []).append(record_index)

    issues = []
    for center, record_indices in records_by_center.items():
        if len(record_indices) < 2:
            continue
        for record_index in record_indices:
            issues.append(
                _gate2_issue(
                    StereoIssueCode.INVALID_ANCHOR,
                    f"atom center row {center} has multiple Level-2 geometry records",
                    record_index=record_index,
                    atom_indices=(center,),
                )
            )
    return tuple(sorted(issues, key=lambda issue: issue.record_index))


def _validate_gate7_coordination_record(
    record: StereoRecord,
    descriptor: _Level2Descriptor,
    *,
    record_index: int,
    atom_count: int,
    adjacency: Sequence[frozenset[int]],
) -> tuple[_Gate7CoordinationSite | None, tuple[StereoValidationIssue, ...]]:
    """Validate one declared coordination site without enabling public Gate 7."""

    family = descriptor.family
    family_name = family.value
    record_atoms = _record_atom_indices(record)
    issues = list(
        _record_issues(
            _validate_stereo_anchor(record.anchor, family=family),
            record_index=record_index,
            atom_indices=record.anchor.atom_indices,
        )
    )

    expected_count = _FAMILY_CARRIER_COUNTS[family]
    if len(record.carriers) != expected_count:
        issues.append(
            _gate2_issue(
                StereoIssueCode.INVALID_CARRIER,
                f"expected {expected_count} ordered carriers, got "
                f"{len(record.carriers)}",
                record_index=record_index,
                atom_indices=record_atoms,
            )
        )

    carrier_rows = []
    seen_rows = set()
    carrier_shape_valid = len(record.carriers) == expected_count
    for position, carrier in enumerate(record.carriers):
        if carrier.kind is not StereoCarrierKind.ATOM:
            issues.append(
                _gate2_issue(
                    StereoIssueCode.UNSUPPORTED,
                    f"{family_name} carrier position {position} uses unsupported "
                    f"{carrier.kind.value}; Gate 7 supports atom-point carriers only",
                    record_index=record_index,
                    atom_indices=record_atoms,
                )
            )
            carrier_shape_valid = False
            continue

        atom = carrier.atom
        if (
            not isinstance(atom, int)
            or isinstance(atom, bool)
            or atom < 0
        ):
            issues.append(
                _gate2_issue(
                    StereoIssueCode.INVALID_CARRIER,
                    f"atom carrier at position {position} requires a "
                    "non-negative atom row",
                    record_index=record_index,
                    atom_indices=record_atoms,
                )
            )
            carrier_shape_valid = False
            continue

        carrier_rows.append(atom)
        if atom in seen_rows:
            issues.append(
                _gate2_issue(
                    StereoIssueCode.INVALID_CARRIER,
                    f"duplicate carrier identity at position {position}",
                    record_index=record_index,
                    atom_indices=record_atoms,
                )
            )
            carrier_shape_valid = False
        seen_rows.add(atom)
        if atom >= atom_count:
            carrier_shape_valid = False

    range_issues = _validate_gate2_atom_ranges(
        record,
        record_index=record_index,
        atom_count=atom_count,
    )
    issues.extend(range_issues)

    center = None
    anchor = record.anchor
    if anchor.kind is StereoAnchorKind.ATOM and len(anchor.atom_indices) == 1:
        candidate = anchor.atom_indices[0]
        if (
            isinstance(candidate, int)
            and not isinstance(candidate, bool)
            and 0 <= candidate < atom_count
        ):
            center = candidate

    if center is not None and center in seen_rows:
        issues.append(
            _gate2_issue(
                StereoIssueCode.INVALID_CARRIER,
                "coordination center cannot be a carrier",
                record_index=record_index,
                atom_indices=(center,),
            )
        )
        carrier_shape_valid = False

    if record.group is not None:
        issues.append(
            _gate2_issue(
                StereoIssueCode.UNSUPPORTED,
                f"{family_name} enhanced-stereo groups are unsupported for "
                "coordination records",
                record_index=record_index,
                atom_indices=record_atoms,
            )
        )

    if center is not None and carrier_shape_valid:
        represented_shell = set(adjacency[center])
        declared_shell = set(carrier_rows)
        if declared_shell != represented_shell:
            represented_atom_rows = {
                row for row in represented_shell if row < atom_count
            }
            issues.append(
                _gate2_issue(
                    StereoIssueCode.INVALID_CARRIER,
                    f"{family_name} carriers must exactly cover the represented "
                    f"first shell at center row {center}",
                    record_index=record_index,
                    atom_indices=(
                        center,
                        *sorted(declared_shell | represented_atom_rows),
                    ),
                )
            )

    if issues:
        return None, tuple(issues)
    return _gate7_site_from_record(record_index, record, descriptor), ()


def _gate7_carrier_components(
    site: _Gate7CoordinationSite,
    adjacency: Sequence[frozenset[int]],
) -> tuple[frozenset[int], ...]:
    """Return carrier-containing components after removing this site's spokes."""

    removed_edges = {
        frozenset((site.center, carrier)) for carrier in site.carrier_rows
    }
    visited = set()
    components = []
    for carrier in sorted(site.carrier_rows):
        if carrier in visited:
            continue
        component = set()
        pending = [carrier]
        visited.add(carrier)
        while pending:
            current = pending.pop()
            component.add(current)
            for neighbour in adjacency[current]:
                if frozenset((current, neighbour)) in removed_edges:
                    continue
                if neighbour not in visited:
                    visited.add(neighbour)
                    pending.append(neighbour)
        components.append(frozenset(component))
    return tuple(components)


def _gate7_chelate_pairs(
    site: _Gate7CoordinationSite,
    adjacency: Sequence[frozenset[int]],
) -> frozenset[tuple[int, int]]:
    """Return the traversal-independent carrier-pair chelate relation."""

    components = _gate7_carrier_components(site, adjacency)
    component_by_carrier = {
        carrier: component_index
        for component_index, component in enumerate(components)
        for carrier in site.carrier_rows
        if carrier in component
    }
    return frozenset(
        (first, second)
        for first, second in combinations(sorted(site.carrier_rows), 2)
        if component_by_carrier[first] == component_by_carrier[second]
    )


def _gate7_coupled_multicenter_issues(
    sites: Sequence[_Gate7CoordinationSite],
    adjacency: Sequence[frozenset[int]],
) -> tuple[StereoValidationIssue, ...]:
    """Diagnose coupled base-valid sites after same-center conflict filtering."""

    coupled = set()
    centers = {site.center: position for position, site in enumerate(sites)}
    for position, site in enumerate(sites):
        for component in _gate7_carrier_components(site, adjacency):
            if site.center in component:
                coupled.add(position)
            for center in component & centers.keys():
                other_position = centers[center]
                if other_position != position:
                    coupled.update((position, other_position))

    return tuple(
        _gate2_issue(
            StereoIssueCode.UNSUPPORTED,
            f"{site.descriptor.family.value} chelate partition at center row "
            f"{site.center} participates in coupled multicenter coordination, "
            "which is unsupported",
            record_index=site.record_index,
            atom_indices=(site.center, *sorted(site.carrier_rows)),
        )
        for position, site in sorted(
            enumerate(sites),
            key=lambda item: item[1].record_index,
        )
        if position in coupled
    )


def _gate7_site_from_record(
    record_index: int,
    record: StereoRecord,
    descriptor: _Level2Descriptor,
) -> _Gate7CoordinationSite:
    """Build a coordination site from a record accepted by Gate 7."""

    return _Gate7CoordinationSite(
        record_index=record_index,
        descriptor=descriptor,
        center=record.anchor.atom_indices[0],
        carrier_rows=tuple(carrier.atom for carrier in record.carriers),
    )


def _gate7_mapped_chelate_pairs(
    pairs: Collection[tuple[int, int]],
    atom_mapping: Mapping[int, int],
) -> frozenset[tuple[int, int]]:
    """Map an atom-identity chelate relation into target row space."""

    return frozenset(
        tuple(sorted((atom_mapping[first], atom_mapping[second])))
        for first, second in pairs
    )


def _validate_gate7_coordination_semantics(
    mol: Molecule,
    payload: StereoPayload,
    descriptors: Mapping[int, _Level2Descriptor],
    blocked_record_indices: Collection[int],
    *,
    projection_result: tuple[
        _Gate6Graph | None, tuple[StereoValidationIssue, ...]
    ]
    | None = None,
) -> tuple[StereoValidationIssue, ...]:
    """Validate declared coordination sites on one represented-graph snapshot."""

    if not any(
        _declares_gate7_coordination(record.descriptor)
        for record in payload.records
    ):
        return ()

    coordination_indices = tuple(
        record_index
        for record_index, descriptor in descriptors.items()
        if descriptor.family in _COORDINATION_FAMILIES
    )
    graph, projection_issues = (
        _gate6_projection(mol)
        if projection_result is None
        else projection_result
    )
    issues = list(projection_issues)
    if graph is None:
        return tuple(issues)

    atom_count = len(mol.atoms)
    adjacency = _gate7_represented_adjacency(graph)
    sites = []
    blocked_record_indices = set(blocked_record_indices)
    for record_index in coordination_indices:
        if record_index in blocked_record_indices:
            continue
        site, record_issues = _validate_gate7_coordination_record(
            payload.records[record_index],
            descriptors[record_index],
            record_index=record_index,
            atom_count=atom_count,
            adjacency=adjacency,
        )
        issues.extend(record_issues)
        if site is not None:
            sites.append(site)
    issues.extend(_gate7_coupled_multicenter_issues(sites, adjacency))
    return tuple(issues)


def _validate_gate2_stereochemistry(
    mol: Molecule,
    *,
    projection_result: tuple[
        _Gate6Graph | None, tuple[StereoValidationIssue, ...]
    ]
    | None = None,
) -> StereoValidationReport:
    """Validate supported manual graph semantics through Level-2 Gate 7."""

    atom_count, adjacency, bond_orders = _molecule_stereo_topology(mol)
    bond_kinds = _molecule_stereo_bond_kinds(mol)
    payload = get_stereo_payload(mol)
    parsed_records = tuple(
        _parse_level2_descriptor(record.descriptor)
        for record in payload.records
    )
    descriptors = {
        record_index: parsed.descriptor
        for record_index, parsed in enumerate(parsed_records)
        if parsed.descriptor is not None and not parsed.issues
    }
    same_center_issues = _gate7_same_center_issues(
        payload,
        descriptors,
        atom_count=atom_count,
    )
    blocked_record_indices = {
        issue.record_index for issue in same_center_issues
    }
    issues = list(same_center_issues)
    seen_anchors = set()
    for record_index, (record, parsed) in enumerate(
        zip(payload.records, parsed_records)
    ):
        record_atoms = _record_atom_indices(record)
        if parsed.issues:
            issues.extend(
                _record_issues(
                    parsed.issues,
                    record_index=record_index,
                    atom_indices=record_atoms,
                )
            )
        if parsed.descriptor is None or parsed.issues:
            continue
        if record_index in blocked_record_indices:
            continue

        family = parsed.descriptor.family
        if family in _COORDINATION_FAMILIES:
            continue
        if family not in (
            _Level2DescriptorFamily.TETRAHEDRAL,
            _Level2DescriptorFamily.DOUBLE_BOND,
            _Level2DescriptorFamily.AXIAL,
        ):
            issues.append(
                _gate2_issue(
                    StereoIssueCode.UNSUPPORTED,
                    f"{family.value} semantics are not implemented in Level-2 Gate 2",
                    record_index=record_index,
                    atom_indices=record_atoms,
                )
            )
            continue

        if (
            record.group is not None
            and family is _Level2DescriptorFamily.DOUBLE_BOND
        ):
            issues.append(
                _gate2_issue(
                    StereoIssueCode.UNSUPPORTED,
                    "DB enhanced-stereo group semantics are unsupported",
                    record_index=record_index,
                    atom_indices=record_atoms,
                )
            )

        anchor_issues = _validate_stereo_anchor(record.anchor, family=family)
        issues.extend(
            _record_issues(
                anchor_issues,
                record_index=record_index,
                atom_indices=record.anchor.atom_indices,
            )
        )
        carrier_issues = (
            _validate_stereo_carriers(record.carriers, expected_count=4)
            if family is _Level2DescriptorFamily.TETRAHEDRAL
            else (
                _validate_gate2_double_bond_carriers(record.carriers)
                if family is _Level2DescriptorFamily.DOUBLE_BOND
                else _validate_gate5_axial_carriers(record.carriers)
            )
        )
        issues.extend(
            _record_issues(
                carrier_issues,
                record_index=record_index,
                atom_indices=record_atoms,
            )
        )
        range_issues = _validate_gate2_atom_ranges(
            record,
            record_index=record_index,
            atom_count=atom_count,
        )
        issues.extend(range_issues)

        if not anchor_issues and not any(
            issue.code is StereoIssueCode.INVALID_ANCHOR
            for issue in range_issues
        ):
            if family in (
                _Level2DescriptorFamily.DOUBLE_BOND,
                _Level2DescriptorFamily.AXIAL,
            ):
                anchor_identity = (
                    family,
                    frozenset(record.anchor.atom_indices)
                    if record.anchor.kind is StereoAnchorKind.BOND
                    else min(
                        record.anchor.atom_indices,
                        record.anchor.atom_indices[::-1],
                    )
                )
                if anchor_identity in seen_anchors:
                    issues.append(
                        _gate2_issue(
                            StereoIssueCode.INVALID_ANCHOR,
                            "duplicate Level-2 stereo anchor",
                            record_index=record_index,
                            atom_indices=record.anchor.atom_indices,
                        )
                    )
                seen_anchors.add(anchor_identity)

            if family is _Level2DescriptorFamily.TETRAHEDRAL:
                issues.extend(
                    _validate_gate2_tetrahedral_graph(
                        record,
                        record_index=record_index,
                        atom_count=atom_count,
                        adjacency=adjacency,
                    )
                )
            elif family is _Level2DescriptorFamily.DOUBLE_BOND:
                issues.extend(
                    _validate_gate2_double_bond_graph(
                        record,
                        record_index=record_index,
                        atom_count=atom_count,
                        adjacency=adjacency,
                        bond_orders=bond_orders,
                    )
                )
            else:
                issues.extend(
                    _validate_gate5_axial_graph(
                        record,
                        record_index=record_index,
                        atom_count=atom_count,
                        adjacency=adjacency,
                        bond_orders=bond_orders,
                        bond_kinds=bond_kinds,
                    )
                )

    issues.extend(
        _validate_gate7_coordination_semantics(
            mol,
            payload,
            descriptors,
            blocked_record_indices,
            projection_result=projection_result,
        )
    )

    return StereoValidationReport(is_valid=not issues, issues=tuple(issues))


def _geometry_issue(
    message: str,
    *,
    record_index: int | None = None,
    atom_indices: tuple[int, ...] = (),
) -> StereoValidationIssue:
    return StereoValidationIssue(
        code=StereoIssueCode.DEGENERATE_GEOMETRY,
        message=message,
        record_index=record_index,
        atom_indices=atom_indices,
    )


def _finite_coordinate_array(
    mol: Molecule,
    coordinates: NDArray[np.float64],
) -> tuple[NDArray[np.float64] | None, StereoValidationIssue | None]:
    """Return a validated coordinate view with the frozen molecule row shape."""

    array = np.asarray(coordinates, dtype=np.float64)
    expected_shape = (len(mol.atoms), 3)
    if array.shape != expected_shape:
        return None, _geometry_issue(
            f"coordinate array must have shape {expected_shape}, got {array.shape}"
        )
    if not np.all(np.isfinite(array)):
        nonfinite_rows = tuple(
            int(row) for row in np.flatnonzero(~np.all(np.isfinite(array), axis=1))
        )
        return None, _geometry_issue(
            "coordinate array contains non-finite values",
            atom_indices=nonfinite_rows,
        )
    return array, None


def _normalized_geometry_sign(
    value: float,
    scale: float,
) -> int:
    """Classify a signed measure using a scale-free deterministic tolerance."""

    if scale == 0.0:
        return 0
    normalized = value / scale
    if abs(normalized) <= _GEOMETRY_RELATIVE_TOLERANCE:
        return 0
    return 1 if normalized > 0.0 else -1


def _is_relatively_degenerate(length: float, local_scale: float) -> bool:
    """Return whether a local length vanishes at the geometry tolerance."""

    return local_scale == 0.0 or length / local_scale <= _GEOMETRY_RELATIVE_TOLERANCE


def _gate7_ideal_carrier_directions(
    descriptor: _Level2Descriptor,
) -> NDArray[np.float64]:
    """Return one ideal configuration in stored carrier order."""

    ideal_sites = _COORDINATION_IDEAL_DIRECTIONS[descriptor.family]
    representative = _COORDINATION_REPRESENTATIVES[descriptor.family][
        int(descriptor.value) - 1
    ]
    directions = np.empty((len(representative), 3), dtype=np.float64)
    for site, carrier_position in enumerate(representative):
        directions[carrier_position] = ideal_sites[site]
    return directions


def _gate7_proper_rotation_fit(
    ideal: NDArray[np.float64],
    observed: NDArray[np.float64],
) -> tuple[NDArray[np.float64], float]:
    """Fit equal-weight unit directions by no-centroid proper Procrustes.

    Kabsch minimizes squared Euclidean chord lengths. The returned score is
    computed afterwards as angular RMS; it is not the optimized objective.
    """

    left, _, right_transpose = np.linalg.svd(ideal.T @ observed)
    correction = np.eye(3, dtype=np.float64)
    correction[-1, -1] = (
        1.0
        if np.linalg.det(left @ right_transpose) >= 0.0
        else -1.0
    )
    rotation = left @ correction @ right_transpose
    fitted = ideal @ rotation
    dots = np.clip(np.sum(fitted * observed, axis=1), -1.0, 1.0)
    cross_norms = np.linalg.norm(np.cross(fitted, observed), axis=1)
    angles = np.arctan2(cross_norms, dots)
    score = float(np.sqrt(np.mean(np.square(angles))))
    return rotation, score


def _gate7_coordination_candidates(
    family: _Level2DescriptorFamily,
    observed: NDArray[np.float64],
) -> tuple[_Gate7CoordinateCandidate, ...]:
    """Rank every descriptor in one coordination family deterministically."""

    candidates = []
    for value in range(1, _COORDINATION_CARDINALITIES[family] + 1):
        descriptor = _Level2Descriptor(family, value)
        ideal = _gate7_ideal_carrier_directions(descriptor)
        _, score = _gate7_proper_rotation_fit(ideal, observed)
        candidates.append(_Gate7CoordinateCandidate(descriptor, score))
    return tuple(
        sorted(
            candidates,
            key=lambda candidate: (
                candidate.score_radians,
                int(candidate.descriptor.value),
            ),
        )
    )


def _select_gate7_coordination_descriptor(
    declared_family: _Level2DescriptorFamily,
    candidates_by_family: Mapping[
        _Level2DescriptorFamily,
        Sequence[_Gate7CoordinateCandidate],
    ],
    *,
    record_index: int,
    atom_indices: tuple[int, ...],
) -> tuple[_Level2Descriptor | None, StereoValidationIssue | None]:
    """Apply the raw-radian 12/2/3-degree policy to ranked candidates."""

    declared = candidates_by_family[declared_family]
    declared_best = declared[0]

    if declared_family in _GATE7_FIVE_COORDINATE_FAMILIES:
        alternate_family = next(
            family
            for family in _GATE7_FIVE_COORDINATE_FAMILIES
            if family is not declared_family
        )
        alternate_best = candidates_by_family[alternate_family][0]
        if (
            declared_best.score_radians > _GATE7_ACCEPTANCE_RADIANS
            and alternate_best.score_radians > _GATE7_ACCEPTANCE_RADIANS
        ):
            return None, _geometry_issue(
                "TBP/SPY coordinate geometry exceeds the 12-degree angular "
                "RMS acceptance cutoff in both families",
                record_index=record_index,
                atom_indices=atom_indices,
            )
        family_gap = abs(
            declared_best.score_radians - alternate_best.score_radians
        )
        if family_gap <= _GATE7_FIVE_COORDINATE_MARGIN_RADIANS:
            return None, _geometry_issue(
                "TBP/SPY coordinate geometry is ambiguous within the "
                "3-degree family margin",
                record_index=record_index,
                atom_indices=atom_indices,
            )
        if alternate_best.score_radians < declared_best.score_radians:
            return None, StereoValidationIssue(
                code=StereoIssueCode.COORDINATE_CONTRADICTION,
                message=(
                    f"declared {declared_family.value} family contradicts "
                    f"coordinate-preferred {alternate_family.value} family"
                ),
                record_index=record_index,
                atom_indices=atom_indices,
            )
    elif declared_best.score_radians > _GATE7_ACCEPTANCE_RADIANS:
        return None, _geometry_issue(
            f"{declared_family.value} coordinate geometry exceeds the "
            "12-degree angular RMS acceptance cutoff",
            record_index=record_index,
            atom_indices=atom_indices,
        )

    if (
        declared[1].score_radians - declared_best.score_radians
        <= _GATE7_SAME_FAMILY_MARGIN_RADIANS
    ):
        return None, _geometry_issue(
            f"{declared_family.value} coordinate geometry is ambiguous within "
            "the 2-degree same-family margin",
            record_index=record_index,
            atom_indices=atom_indices,
        )
    return declared_best.descriptor, None


def _gate7_center_rooted_directions(
    site: _Gate7CoordinationSite,
    coordinates: NDArray[np.float64],
) -> tuple[NDArray[np.float64] | None, StereoValidationIssue | None]:
    """Normalize each center-to-carrier vector without donor centering."""

    rows = (site.center, *site.carrier_rows)
    site_coordinates = coordinates[np.asarray(rows)]
    coordinate_scale = float(np.max(np.abs(site_coordinates)))
    scaled_coordinates = np.zeros_like(site_coordinates, dtype=np.float64)
    if coordinate_scale != 0.0:
        scaled_coordinates = site_coordinates / coordinate_scale
    vectors = scaled_coordinates[1:] - scaled_coordinates[0]
    component_scales = np.max(np.abs(vectors), axis=1)
    scaled_vectors = np.zeros_like(vectors, dtype=np.float64)
    np.divide(
        vectors,
        component_scales[:, np.newaxis],
        out=scaled_vectors,
        where=component_scales[:, np.newaxis] != 0.0,
    )
    scaled_norms = np.linalg.norm(scaled_vectors, axis=1)
    max_component_scale = float(np.max(component_scales))
    relative_norms = np.zeros_like(component_scales, dtype=np.float64)
    if max_component_scale != 0.0:
        relative_norms = (
            component_scales / max_component_scale
        ) * scaled_norms
    local_scale = float(np.max(relative_norms))
    if any(
        _is_relatively_degenerate(float(norm), local_scale)
        for norm in relative_norms
    ):
        return None, _geometry_issue(
            f"{site.descriptor.family.value} coordinate geometry has a carrier "
            "coincident with its center within relative tolerance "
            f"{_GEOMETRY_RELATIVE_TOLERANCE:g}",
            record_index=site.record_index,
            atom_indices=(site.center, *site.carrier_rows),
        )
    directions = scaled_vectors / scaled_norms[:, np.newaxis]
    return directions, None


def _gate7_coordination_coordinate_descriptor(
    site: _Gate7CoordinationSite,
    coordinates: NDArray[np.float64],
) -> tuple[_Level2Descriptor | None, StereoValidationIssue | None]:
    """Assign one private Gate-7 descriptor without public cutover."""

    directions, issue = _gate7_center_rooted_directions(site, coordinates)
    if issue is not None:
        return None, issue

    family = site.descriptor.family
    families = [family]
    if family in _GATE7_FIVE_COORDINATE_FAMILIES:
        families.extend(
            candidate_family
            for candidate_family in _GATE7_FIVE_COORDINATE_FAMILIES
            if candidate_family is not family
        )
    candidates = {
        candidate_family: _gate7_coordination_candidates(
            candidate_family,
            directions,
        )
        for candidate_family in families
    }
    return _select_gate7_coordination_descriptor(
        family,
        candidates,
        record_index=site.record_index,
        atom_indices=(site.center, *site.carrier_rows),
    )


def _tetrahedral_coordinate_descriptor(
    record: StereoRecord,
    coordinates: NDArray[np.float64],
    *,
    record_index: int,
) -> tuple[_Level2Descriptor | None, StereoValidationIssue | None]:
    """Assign TH parity from the ordered-carrier signed tetrahedral volume.

    Positive normalized volume is ``even`` and negative volume is ``odd``.
    An implicit carrier uses the anchor as an orientation-only zero-vector
    proxy.  This preserves ordered-carrier parity without inventing a bond
    length or direction; three explicit bonds coplanar with the anchor remain
    correctly unresolved.
    """

    center = record.anchor.atom_indices[0]
    center_coordinate = coordinates[center]
    vectors = []
    explicit_vectors = []
    for carrier in record.carriers:
        if carrier.kind is StereoCarrierKind.ATOM:
            vector = coordinates[carrier.atom] - center_coordinate
            vectors.append(vector)
            explicit_vectors.append(vector)
        else:
            vectors.append(np.zeros(3, dtype=np.float64))

    explicit_norms = tuple(float(np.linalg.norm(vector)) for vector in explicit_vectors)
    local_scale = max(explicit_norms, default=0.0)
    if any(_is_relatively_degenerate(norm, local_scale) for norm in explicit_norms):
        return None, _geometry_issue(
            "TH coordinate geometry has a carrier coincident with its center "
            "within relative tolerance "
            f"{_GEOMETRY_RELATIVE_TOLERANCE:g}",
            record_index=record_index,
            atom_indices=_record_atom_indices(record),
        )

    carrier_coordinates = tuple(center_coordinate + vector for vector in vectors)
    characteristic_length = max(
        float(np.linalg.norm(carrier_coordinates[left] - carrier_coordinates[right]))
        for left in range(4)
        for right in range(left)
    )
    if characteristic_length == 0.0:
        sign = 0
    else:
        first = (
            carrier_coordinates[0] - carrier_coordinates[3]
        ) / characteristic_length
        second = (
            carrier_coordinates[1] - carrier_coordinates[3]
        ) / characteristic_length
        third = (
            carrier_coordinates[2] - carrier_coordinates[3]
        ) / characteristic_length
        signed_volume = float(np.dot(first, np.cross(second, third)))
        sign = _normalized_geometry_sign(signed_volume, 1.0)
    if sign == 0:
        return None, _geometry_issue(
            "TH signed volume is degenerate within relative tolerance "
            f"{_GEOMETRY_RELATIVE_TOLERANCE:g}",
            record_index=record_index,
            atom_indices=_record_atom_indices(record),
        )
    return (
        _Level2Descriptor(
            _Level2DescriptorFamily.TETRAHEDRAL,
            "even" if sign > 0 else "odd",
        ),
        None,
    )


def _double_bond_coordinate_descriptor(
    record: StereoRecord,
    coordinates: NDArray[np.float64],
    *,
    record_index: int,
) -> tuple[_Level2Descriptor | None, StereoValidationIssue | None]:
    """Assign DB relation from endpoint-scoped ordered transverse vectors."""

    begin, end = record.anchor.atom_indices
    axis = coordinates[end] - coordinates[begin]
    axis_norm = float(np.linalg.norm(axis))
    endpoint_vectors = tuple(
        tuple(
            coordinates[carrier.atom] - coordinates[endpoint]
            for carrier in record.carriers[2 * endpoint_index : 2 * endpoint_index + 2]
            if carrier.kind is StereoCarrierKind.ATOM
        )
        for endpoint_index, endpoint in enumerate((begin, end))
    )
    explicit_norms = tuple(
        float(np.linalg.norm(vector))
        for vectors in endpoint_vectors
        for vector in vectors
    )
    local_scale = max((axis_norm, *explicit_norms))
    if _is_relatively_degenerate(axis_norm, local_scale):
        return None, _geometry_issue(
            "DB coordinate geometry has coincident bond endpoints within "
            f"relative tolerance {_GEOMETRY_RELATIVE_TOLERANCE:g}",
            record_index=record_index,
            atom_indices=_record_atom_indices(record),
        )
    if any(_is_relatively_degenerate(norm, local_scale) for norm in explicit_norms):
        return None, _geometry_issue(
            "DB coordinate geometry has a carrier coincident with its endpoint "
            "within relative tolerance "
            f"{_GEOMETRY_RELATIVE_TOLERANCE:g}",
            record_index=record_index,
            atom_indices=_record_atom_indices(record),
        )
    axis_unit = axis / axis_norm

    endpoint_directions = []
    for endpoint_index, endpoint in enumerate((begin, end)):
        transverse = []
        group = record.carriers[2 * endpoint_index : 2 * endpoint_index + 2]
        for carrier in group:
            if carrier.kind is StereoCarrierKind.ATOM:
                vector = coordinates[carrier.atom] - coordinates[endpoint]
                transverse.append(vector - np.dot(vector, axis_unit) * axis_unit)
            else:
                transverse.append(np.zeros(3, dtype=np.float64))

        direction = transverse[0] - transverse[1]
        direction_norm = float(np.linalg.norm(direction))
        endpoint_scale = max(
            (
                axis_norm,
                *(
                    float(np.linalg.norm(vector))
                    for vector in endpoint_vectors[endpoint_index]
                ),
            )
        )
        if _is_relatively_degenerate(direction_norm, endpoint_scale):
            return None, _geometry_issue(
                f"DB endpoint {endpoint} has degenerate transverse carriers "
                "within relative tolerance "
                f"{_GEOMETRY_RELATIVE_TOLERANCE:g}",
                record_index=record_index,
                atom_indices=_record_atom_indices(record),
            )
        endpoint_directions.append(direction)

    relation = float(
        np.dot(
            endpoint_directions[0] / np.linalg.norm(endpoint_directions[0]),
            endpoint_directions[1] / np.linalg.norm(endpoint_directions[1]),
        )
    )
    sign = _normalized_geometry_sign(relation, 1.0)
    if sign == 0:
        return None, _geometry_issue(
            "DB transverse relation is degenerate within relative tolerance "
            f"{_GEOMETRY_RELATIVE_TOLERANCE:g}",
            record_index=record_index,
            atom_indices=_record_atom_indices(record),
        )
    return (
        _Level2Descriptor(
            _Level2DescriptorFamily.DOUBLE_BOND,
            "together" if sign > 0 else "opposite",
        ),
        None,
    )


def _axial_coordinate_descriptor(
    record: StereoRecord,
    coordinates: NDArray[np.float64],
    *,
    record_index: int,
) -> tuple[_Level2Descriptor | None, StereoValidationIssue | None]:
    """Assign AX parity from its ordered endpoint carrier-pair triple product."""

    begin = record.anchor.atom_indices[0]
    end = record.anchor.atom_indices[-1]
    axis = coordinates[end] - coordinates[begin]
    axis_norm = float(np.linalg.norm(axis))
    endpoint_vectors = tuple(
        tuple(
            coordinates[carrier.atom] - coordinates[endpoint]
            for carrier in record.carriers[
                2 * endpoint_index : 2 * endpoint_index + 2
            ]
            if carrier.kind is StereoCarrierKind.ATOM
        )
        for endpoint_index, endpoint in enumerate((begin, end))
    )
    explicit_norms = tuple(
        float(np.linalg.norm(vector))
        for vectors in endpoint_vectors
        for vector in vectors
    )
    local_scale = max((axis_norm, *explicit_norms))
    if _is_relatively_degenerate(axis_norm, local_scale):
        return None, _geometry_issue(
            "AX coordinate geometry has coincident axis endpoints within "
            f"relative tolerance {_GEOMETRY_RELATIVE_TOLERANCE:g}",
            record_index=record_index,
            atom_indices=_record_atom_indices(record),
        )
    axis_unit = axis / axis_norm

    endpoint_directions = []
    for endpoint_index, endpoint in enumerate((begin, end)):
        explicit_at_endpoint = endpoint_vectors[endpoint_index]
        endpoint_scale = max(
            (
                axis_norm,
                *(
                    float(np.linalg.norm(vector))
                    for vector in explicit_at_endpoint
                ),
            )
        )
        if any(
            _is_relatively_degenerate(
                float(np.linalg.norm(vector)),
                endpoint_scale,
            )
            for vector in explicit_at_endpoint
        ):
            return None, _geometry_issue(
                "AX coordinate geometry has a carrier coincident with its endpoint "
                "within relative tolerance "
                f"{_GEOMETRY_RELATIVE_TOLERANCE:g}",
                record_index=record_index,
                atom_indices=_record_atom_indices(record),
            )

        transverse = []
        group = record.carriers[2 * endpoint_index : 2 * endpoint_index + 2]
        for carrier in group:
            if carrier.kind is StereoCarrierKind.ATOM:
                vector = coordinates[carrier.atom] - coordinates[endpoint]
                transverse.append(vector - np.dot(vector, axis_unit) * axis_unit)
            else:
                transverse.append(np.zeros(3, dtype=np.float64))

        direction = transverse[0] - transverse[1]
        direction_norm = float(np.linalg.norm(direction))
        if _is_relatively_degenerate(direction_norm, endpoint_scale):
            return None, _geometry_issue(
                f"AX endpoint {endpoint} has degenerate projected carriers "
                "within relative tolerance "
                f"{_GEOMETRY_RELATIVE_TOLERANCE:g}",
                record_index=record_index,
                atom_indices=_record_atom_indices(record),
            )
        endpoint_directions.append(direction)

    handedness = float(
        np.dot(
            axis_unit,
            np.cross(
                endpoint_directions[0] / np.linalg.norm(endpoint_directions[0]),
                endpoint_directions[1] / np.linalg.norm(endpoint_directions[1]),
            ),
        )
    )
    sign = _normalized_geometry_sign(handedness, 1.0)
    if sign == 0:
        return None, _geometry_issue(
            "AX endpoint carrier triple product is degenerate within relative "
            f"tolerance {_GEOMETRY_RELATIVE_TOLERANCE:g}",
            record_index=record_index,
            atom_indices=_record_atom_indices(record),
        )
    return (
        _Level2Descriptor(
            _Level2DescriptorFamily.AXIAL,
            "even" if sign > 0 else "odd",
        ),
        None,
    )


def _gate4_coordinate_assignments(
    mol: Molecule,
    coordinates: NDArray[np.float64],
    graph_report: StereoValidationReport,
) -> tuple[dict[int, _Level2Descriptor], tuple[StereoValidationIssue, ...]]:
    """Assign coordinates only for records accepted by the earlier gates."""

    if any(issue.record_index is None for issue in graph_report.issues):
        return {}, ()
    array, array_issue = _finite_coordinate_array(mol, coordinates)
    if array_issue is not None:
        return {}, (array_issue,)

    invalid_records = {
        issue.record_index
        for issue in graph_report.issues
        if issue.record_index is not None
    }
    assignments = {}
    issues = []
    for record_index, record in enumerate(get_stereo_payload(mol).records):
        if record_index in invalid_records:
            continue
        parsed = _parse_level2_descriptor(record.descriptor).descriptor
        if parsed is None or parsed.family not in (
            _Level2DescriptorFamily.TETRAHEDRAL,
            _Level2DescriptorFamily.DOUBLE_BOND,
            _Level2DescriptorFamily.AXIAL,
            *_COORDINATION_FAMILIES,
        ):
            continue
        if parsed.family is _Level2DescriptorFamily.TETRAHEDRAL:
            assigned, geometry_issue = _tetrahedral_coordinate_descriptor(
                record,
                array,
                record_index=record_index,
            )
        elif parsed.family is _Level2DescriptorFamily.DOUBLE_BOND:
            assigned, geometry_issue = _double_bond_coordinate_descriptor(
                record,
                array,
                record_index=record_index,
            )
        elif parsed.family is _Level2DescriptorFamily.AXIAL:
            assigned, geometry_issue = _axial_coordinate_descriptor(
                record,
                array,
                record_index=record_index,
            )
        else:
            assigned, geometry_issue = _gate7_coordination_coordinate_descriptor(
                _gate7_site_from_record(record_index, record, parsed),
                array,
            )
        if geometry_issue is not None:
            issues.append(geometry_issue)
            continue
        assignments[record_index] = assigned

    issues.extend(
        _coordinate_group_consistency_issues(
            get_stereo_payload(mol),
            assignments,
        )
    )
    return assignments, tuple(issues)


def _records_atom_union(
    records: Sequence[StereoRecord],
    indices: Collection[int],
) -> tuple[int, ...]:
    """Return stable atom context for one enhanced-stereo relation."""

    return tuple(
        sorted(
            {
                atom
                for index in indices
                for atom in _record_atom_indices(records[index])
            }
        )
    )


def _coordinate_group_consistency_issues(
    payload: StereoPayload,
    assignments: Mapping[int, _Level2Descriptor],
) -> tuple[StereoValidationIssue, ...]:
    """Compare raw coordinate assignments with fixed and relative records."""

    issues = []
    relative_groups = {}
    for record_index, record in enumerate(payload.records):
        group = record.group
        if group is not None and group.kind in (
            StereoGroupKind.AND,
            StereoGroupKind.OR,
        ):
            relative_groups.setdefault(group, []).append(record_index)
            continue
        if record_index not in assignments:
            continue
        parsed = _parse_level2_descriptor(record.descriptor).descriptor
        assigned = assignments[record_index]
        if assigned != parsed:
            issues.append(
                StereoValidationIssue(
                    code=StereoIssueCode.COORDINATE_CONTRADICTION,
                    message=(
                        f"stored descriptor {record.descriptor} contradicts "
                        "coordinate descriptor "
                        f"{_render_level2_descriptor(assigned)}"
                    ),
                    record_index=record_index,
                    atom_indices=_record_atom_indices(record),
                )
            )

    for group, member_indices in relative_groups.items():
        assigned_members = tuple(
            index for index in member_indices if index in assignments
        )
        if len(assigned_members) != len(member_indices):
            continue
        inversions = {
            assignments[index]
            != _parse_level2_descriptor(payload.records[index].descriptor).descriptor
            for index in assigned_members
        }
        if len(inversions) <= 1:
            continue
        reference_index = min(member_indices)
        issues.append(
            StereoValidationIssue(
                code=StereoIssueCode.COORDINATE_CONTRADICTION,
                message=(
                    f"{group.kind.value} enhanced-stereo group has a partial "
                    "coordinate inversion"
                ),
                record_index=reference_index,
                atom_indices=_records_atom_union(
                    payload.records,
                    member_indices,
                ),
            )
        )
    return tuple(issues)


def _gate7_projection_result(
    mol: Molecule,
) -> tuple[_Gate6Graph | None, tuple[StereoValidationIssue, ...]] | None:
    """Project once when a parsed coordination declaration requires Gate 7."""

    has_coordination = any(
        _declares_gate7_coordination(record.descriptor)
        for record in get_stereo_payload(mol).records
    )
    return _gate6_projection(mol) if has_coordination else None


def _validate_graph_stereochemistry(
    mol: Molecule,
    *,
    projection_result: tuple[
        _Gate6Graph | None, tuple[StereoValidationIssue, ...]
    ]
    | None = None,
) -> StereoValidationReport:
    """Combine structural Gates 2/5 with exact Gate-6 stereogenicity."""

    if projection_result is None:
        projection_result = _gate7_projection_result(mol)
    report = _validate_gate2_stereochemistry(
        mol,
        projection_result=projection_result,
    )
    issues = _gate6_stereogenicity_issues(
        mol,
        report,
        projection_result=projection_result,
    )
    if not issues:
        return report
    return StereoValidationReport(
        is_valid=False,
        issues=(*report.issues, *issues),
    )


def _direct_priority_order(
    mol: Molecule,
    carriers: Sequence[StereoCarrier],
    atom_facts: Mapping[int, AtomTransportRecord],
) -> tuple[int, ...] | None:
    """Order carriers high-to-low when direct Z/isotope facts are complete."""

    priorities = []
    for carrier in carriers:
        if carrier.kind is StereoCarrierKind.IMPLICIT_H:
            priorities.append((1, 1))
            continue
        atom = mol.atoms[carrier.atom]
        atomic_number = int(atom.atomic_number)
        fact = atom_facts.get(carrier.atom)
        isotope = None if fact is None else fact.isotope
        if atomic_number == 1:
            mass = (
                1
                if isotope in (None, 0, 1)
                else (
                    int(isotope)
                    if int(isotope) > 1
                    else None
                )
            )
        else:
            mass = (
                int(isotope)
                if isotope is not None and int(isotope) > 0
                else None
            )
        priorities.append((atomic_number, mass))

    for atomic_number in {priority[0] for priority in priorities}:
        masses = tuple(
            mass
            for number, mass in priorities
            if number == atomic_number
        )
        if len(masses) > 1 and (
            any(mass is None for mass in masses)
            or len(set(masses)) != len(masses)
        ):
            return None
    return tuple(
        sorted(
            range(len(carriers)),
            key=lambda index: (
                priorities[index][0],
                0 if priorities[index][1] is None else priorities[index][1],
            ),
            reverse=True,
        )
    )


def _direct_priority_assignments(
    mol: Molecule,
    payload: StereoPayload,
    graph_report: StereoValidationReport,
) -> tuple[
    tuple[StereoDerivedDescriptor, ...],
    tuple[StereoValidationIssue, ...],
]:
    """Derive only the Stage-17 direct-priority TH and DB naming subset."""

    if any(issue.record_index is None for issue in graph_report.issues):
        return (), ()
    invalid_records = {
        issue.record_index
        for issue in graph_report.issues
        if issue.record_index is not None
    }
    atom_facts = {fact.atom: fact for fact in payload.atom_facts}
    derived = []
    issues = []
    for record_index, record in enumerate(payload.records):
        if record_index in invalid_records:
            continue
        descriptor = _parse_level2_descriptor(record.descriptor).descriptor
        if descriptor.family is _Level2DescriptorFamily.TETRAHEDRAL:
            order = _direct_priority_order(
                mol,
                record.carriers,
                atom_facts,
            )
            if order is None:
                issues.append(
                    _gate2_issue(
                        StereoIssueCode.UNSUPPORTED,
                        "direct CIP priority does not resolve all TH carriers",
                        record_index=record_index,
                        atom_indices=_record_atom_indices(record),
                    )
                )
                continue
            ordered = replace(
                record,
                carriers=tuple(record.carriers[index] for index in order),
            )
            transformed = _transform_supported_record(
                descriptor,
                record,
                ordered,
            ).descriptor
            label = "S" if transformed.value == "even" else "R"
            stereo_type = StereoType.TETRAHEDRAL
            system = "CIP"
        elif descriptor.family is _Level2DescriptorFamily.DOUBLE_BOND:
            begin_order = _direct_priority_order(
                mol,
                record.carriers[:2],
                atom_facts,
            )
            end_order = _direct_priority_order(
                mol,
                record.carriers[2:],
                atom_facts,
            )
            if begin_order is None or end_order is None:
                issues.append(
                    _gate2_issue(
                        StereoIssueCode.UNSUPPORTED,
                        "direct E/Z priority does not resolve both DB endpoint pairs",
                        record_index=record_index,
                        atom_indices=_record_atom_indices(record),
                    )
                )
                continue
            order = (*begin_order, *(index + 2 for index in end_order))
            ordered = replace(
                record,
                carriers=tuple(record.carriers[index] for index in order),
            )
            transformed = _transform_supported_record(
                descriptor,
                record,
                ordered,
            ).descriptor
            label = "Z" if transformed.value == "together" else "E"
            stereo_type = StereoType.DOUBLE_BOND
            system = "E/Z"
        elif descriptor.family is _Level2DescriptorFamily.AXIAL:
            issues.append(
                _gate2_issue(
                    StereoIssueCode.UNSUPPORTED,
                    "AX axial CIP assignment is outside the Stage-17 "
                    "direct-priority subset",
                    record_index=record_index,
                    atom_indices=_record_atom_indices(record),
                )
            )
            continue
        else:
            continue

        derived.append(
            StereoDerivedDescriptor(
                record_index=record_index,
                stereo_type=stereo_type,
                label=label,
                system=system,
                is_absolute=(
                    record.group is None
                    or record.group.kind is StereoGroupKind.ABS
                ),
            )
        )
    return tuple(derived), tuple(issues)


def validate_stereochemistry(
    mol: Molecule,
    *,
    coordinates: NDArray[np.float64] | None = None,
) -> StereoValidationReport:
    """Validate graph stereogenicity and optional coordinate semantics."""

    graph_report = _validate_graph_stereochemistry(mol)
    if coordinates is None:
        return graph_report
    _, coordinate_issues = _gate4_coordinate_assignments(
        mol,
        coordinates,
        graph_report,
    )
    issues = (*graph_report.issues, *coordinate_issues)
    return StereoValidationReport(is_valid=not issues, issues=issues)


def assign_stereochemistry(
    mol: Molecule,
    *,
    coordinates: NDArray[np.float64] | None = None,
) -> StereoAssignmentReport:
    """Assign coordinate descriptors and direct-priority names without mutation."""

    payload = get_stereo_payload(mol)
    graph_report = _validate_graph_stereochemistry(mol)
    coordinate_issues = ()
    assigned_payload = payload
    if coordinates is not None:
        assignments, coordinate_issues = _gate4_coordinate_assignments(
            mol,
            coordinates,
            graph_report,
        )
        records = tuple(
            replace(
                record,
                descriptor=_render_level2_descriptor(assignments[index]),
            )
            if index in assignments
            else record
            for index, record in enumerate(payload.records)
        )
        assigned_payload = replace(payload, records=records)
    derived, priority_issues = _direct_priority_assignments(
        mol,
        assigned_payload,
        graph_report,
    )
    return StereoAssignmentReport(
        payload=assigned_payload,
        unresolved=(
            *graph_report.issues,
            *coordinate_issues,
            *priority_issues,
        ),
        derived=derived,
    )


def _alignment_validation_issues(
    side: str,
    report: StereoValidationReport,
) -> tuple[StereoValidationIssue, ...]:
    """Identify the molecule owning each pre-alignment validation issue."""

    return tuple(
        replace(issue, message=f"{side}: {issue.message}")
        for issue in report.issues
    )


def _alignment_mapping_issues(
    reference: Molecule,
    target: Molecule,
    atom_mapping: Mapping[int, int],
) -> tuple[tuple[StereoValidationIssue, ...], dict[int, int]]:
    """Validate a one-to-one mapping over all stereo-relevant atom rows."""

    reference_count = len(reference.atoms)
    target_count = len(target.atoms)
    issues = []
    valid_mapping = {}
    for source, destination in sorted(
        atom_mapping.items(),
        key=lambda item: (repr(item[0]), repr(item[1])),
    ):
        if (
            not isinstance(source, int)
            or isinstance(source, bool)
            or not 0 <= source < reference_count
        ):
            issues.append(
                StereoValidationIssue(
                    StereoIssueCode.INCOMPLETE_MAPPING,
                    f"reference atom row {source!r} is outside molecule range "
                    f"0..{reference_count - 1}",
                )
            )
            continue
        if (
            not isinstance(destination, int)
            or isinstance(destination, bool)
            or not 0 <= destination < target_count
        ):
            issues.append(
                StereoValidationIssue(
                    StereoIssueCode.INCOMPLETE_MAPPING,
                    f"target atom row {destination!r} is outside molecule range "
                    f"0..{target_count - 1}",
                    atom_indices=(source,),
                )
            )
            continue
        valid_mapping[source] = destination

    destinations = {}
    for source, destination in sorted(valid_mapping.items()):
        destinations.setdefault(destination, []).append(source)
    for destination, sources in destinations.items():
        if len(sources) > 1:
            issues.append(
                StereoValidationIssue(
                    StereoIssueCode.INCOMPLETE_MAPPING,
                    f"atom mapping is many-to-one at target row {destination}: "
                    f"reference rows {tuple(sources)}",
                    atom_indices=tuple(sources),
                )
            )

    reference_atoms = sorted(
        {
            atom
            for record in get_stereo_payload(reference).records
            for atom in _record_atom_indices(record)
        }
    )
    target_atoms = sorted(
        {
            atom
            for record in get_stereo_payload(target).records
            for atom in _record_atom_indices(record)
        }
    )
    missing_reference = tuple(
        atom for atom in reference_atoms if atom not in valid_mapping
    )
    if missing_reference:
        issues.append(
            StereoValidationIssue(
                StereoIssueCode.INCOMPLETE_MAPPING,
                "atom mapping does not cover reference stereo rows "
                f"{missing_reference}",
                atom_indices=missing_reference,
            )
        )
    mapped_destinations = set(valid_mapping.values())
    missing_target = tuple(
        atom for atom in target_atoms if atom not in mapped_destinations
    )
    if missing_target:
        issues.append(
            StereoValidationIssue(
                StereoIssueCode.INCOMPLETE_MAPPING,
                f"atom mapping does not cover target stereo rows {missing_target}",
                atom_indices=missing_target,
            )
        )
    return tuple(issues), valid_mapping


def _mapped_alignment_record(
    record: StereoRecord,
    atom_mapping: Mapping[int, int],
) -> StereoRecord:
    """Map atom rows without changing carrier order or descriptor meaning."""

    return replace(
        record,
        anchor=replace(
            record.anchor,
            atom_indices=tuple(
                atom_mapping[atom] for atom in record.anchor.atom_indices
            ),
        ),
        carriers=tuple(
            replace(carrier, atom=atom_mapping[carrier.atom])
            if carrier.kind is StereoCarrierKind.ATOM
            else carrier
            for carrier in record.carriers
        ),
    )


def _alignment_record_identity(
    family: _Level2DescriptorFamily,
    anchor: StereoAnchor,
) -> tuple[object, ...]:
    """Return the orientation-independent identity of a Gate-3 element."""

    if family in _ATOM_GEOMETRY_FAMILIES:
        return family, anchor.atom_indices[0]
    if family is _Level2DescriptorFamily.AXIAL:
        oriented_identity = (
            frozenset(anchor.atom_indices)
            if anchor.kind is StereoAnchorKind.BOND
            else min(anchor.atom_indices, anchor.atom_indices[::-1])
        )
        return family, anchor.kind, oriented_identity
    return family, frozenset(anchor.atom_indices)


def _transform_gate3_double_bond(
    descriptor: _Level2Descriptor,
    source: StereoRecord,
    target: StereoRecord,
) -> _Level2AlgebraResult:
    """Transform DB identity with carrier identity scoped to each endpoint.

    Gate 1's DB rule is descriptor XOR permutation parity. Gate 3 applies the
    same rule after matching each two-carrier group to its bond endpoint. This
    endpoint scope permits the same virtual-H identity or ring atom at both
    ends without conflating the two occurrences.
    """

    anchor_reversed = source.anchor.atom_indices == target.anchor.atom_indices[::-1]
    reordering = []
    for target_group in range(2):
        source_group = 1 - target_group if anchor_reversed else target_group
        source_start = 2 * source_group
        source_carriers = source.carriers[source_start : source_start + 2]
        for carrier in target.carriers[2 * target_group : 2 * target_group + 2]:
            identity = carrier.kind, carrier.atom
            positions = tuple(
                source_start + offset
                for offset, candidate in enumerate(source_carriers)
                if (candidate.kind, candidate.atom) == identity
            )
            if len(positions) != 1:
                return _Level2AlgebraResult(
                    None,
                    (
                        _algebra_issue(
                            StereoIssueCode.INVALID_CARRIER,
                            "mapped DB carriers do not match target endpoint groups",
                        ),
                    ),
                )
            reordering.append(positions[0])

    if sorted(reordering) != list(range(4)):
        return _Level2AlgebraResult(
            None,
            (
                _algebra_issue(
                    StereoIssueCode.INVALID_CARRIER,
                    "mapped DB carriers do not match target endpoint groups",
                ),
            ),
        )
    pairs_reversed = frozenset(reordering[:2]) == frozenset((2, 3))
    if anchor_reversed != pairs_reversed:
        return _Level2AlgebraResult(
            None,
            (
                _algebra_issue(
                    StereoIssueCode.INVALID_CARRIER,
                    "bond-anchor reversal and endpoint carrier-pair order disagree",
                ),
            ),
        )

    relation = _ALKENE_VALUES.index(descriptor.value)
    value = _ALKENE_VALUES[relation ^ _permutation_parity(reordering)]
    return _Level2AlgebraResult(
        _Level2Descriptor(_Level2DescriptorFamily.DOUBLE_BOND, value),
        (),
    )


def _transform_gate5_axial(
    descriptor: _Level2Descriptor,
    source: StereoRecord,
    target: StereoRecord,
) -> _Level2AlgebraResult:
    """Transform AX parity using endpoint-scoped carrier identity."""

    if source.anchor.kind is not target.anchor.kind:
        return _Level2AlgebraResult(
            None,
            (
                _algebra_issue(
                    StereoIssueCode.INVALID_ANCHOR,
                    "AXIS and BOND axial anchors cannot cross-match",
                ),
            ),
        )
    anchor_reversed = source.anchor.atom_indices == target.anchor.atom_indices[::-1]
    if target.anchor.atom_indices not in (
        source.anchor.atom_indices,
        source.anchor.atom_indices[::-1],
    ):
        return _Level2AlgebraResult(
            None,
            (
                _algebra_issue(
                    StereoIssueCode.INVALID_ANCHOR,
                    "AX anchor must preserve identity or fully reverse",
                ),
            ),
        )
    reordering = _endpoint_pair_reordering(
        source.carriers,
        target.carriers,
        endpoint_groups_reversed=anchor_reversed,
    )
    if reordering is None:
        return _Level2AlgebraResult(
            None,
            (
                _algebra_issue(
                    StereoIssueCode.INVALID_CARRIER,
                    "mapped AX carriers do not match target endpoint groups",
                ),
            ),
        )
    parity = _PARITY_VALUES.index(descriptor.value)
    value = _PARITY_VALUES[parity ^ _permutation_parity(reordering)]
    return _Level2AlgebraResult(
        _Level2Descriptor(_Level2DescriptorFamily.AXIAL, value),
        (),
    )


@dataclass(frozen=True)
class _AlignmentMatch:
    """One successfully transformed source/target stereo-record pair."""

    reference_index: int
    target_index: int
    mapped: StereoRecord
    target: StereoRecord
    transformed: _Level2Descriptor


def _relative_group(record: StereoRecord) -> StereoGroupRef | None:
    group = record.group
    if group is not None and group.kind in (
        StereoGroupKind.AND,
        StereoGroupKind.OR,
    ):
        return group
    return None


def _alignment_group_issues(
    reference_payload: StereoPayload,
    target_payload: StereoPayload,
    matches: Sequence[_AlignmentMatch],
) -> tuple[StereoValidationIssue, ...]:
    """Compare fixed descriptors and mapped enhanced-group partitions."""

    issues = []
    matched_by_reference = {
        match.reference_index: match for match in matches
    }
    matched_target_indices = {match.target_index for match in matches}
    source_members = {}
    target_members = {}
    for index, record in enumerate(reference_payload.records):
        group = _relative_group(record)
        if group is not None:
            source_members.setdefault(group, set()).add(index)
    for index, record in enumerate(target_payload.records):
        group = _relative_group(record)
        if group is not None:
            target_members.setdefault(group, set()).add(index)

    adjacency = {}
    match_nodes = []
    for match in matches:
        source_group = _relative_group(
            reference_payload.records[match.reference_index]
        )
        target_group = _relative_group(match.target)
        if source_group is None and target_group is None:
            target_descriptor = _parse_level2_descriptor(
                match.target.descriptor
            ).descriptor
            if match.transformed != target_descriptor:
                issues.append(
                    StereoValidationIssue(
                        StereoIssueCode.INVALID_DESCRIPTOR,
                        "mapped descriptor "
                        f"{_render_level2_descriptor(match.transformed)} does "
                        "not match target descriptor "
                        f"{match.target.descriptor}",
                        record_index=match.reference_index,
                        atom_indices=_record_atom_indices(match.mapped),
                    )
                )
            continue

        source_node = (
            ("source_group", source_group)
            if source_group is not None
            else ("source_fixed", match.reference_index)
        )
        target_node = (
            ("target_group", target_group)
            if target_group is not None
            else ("target_fixed", match.target_index)
        )
        adjacency.setdefault(source_node, set()).add(target_node)
        adjacency.setdefault(target_node, set()).add(source_node)
        match_nodes.append((match, source_node, target_node))

    components = []
    unseen = set(adjacency)
    while unseen:
        seed = min(unseen, key=repr)
        component = set()
        pending = [seed]
        while pending:
            node = pending.pop()
            if node in component:
                continue
            component.add(node)
            pending.extend(adjacency[node] - component)
        unseen.difference_update(component)
        component_matches = tuple(
            match
            for match, source_node, target_node in match_nodes
            if source_node in component or target_node in component
        )
        components.append((component, component_matches))

    components.sort(
        key=lambda item: min(
            match.reference_index for match in item[1]
        )
    )
    for component, component_matches in components:
        source_groups = tuple(
            node[1] for node in component if node[0] == "source_group"
        )
        target_groups = tuple(
            node[1] for node in component if node[0] == "target_group"
        )
        reference_indices = set()
        target_indices = set()
        for node_kind, value in component:
            if node_kind == "source_group":
                reference_indices.update(source_members[value])
            elif node_kind == "source_fixed":
                reference_indices.add(value)
            elif node_kind == "target_group":
                target_indices.update(target_members[value])
            else:
                target_indices.add(value)

        if (
            not reference_indices <= matched_by_reference.keys()
            or not target_indices <= matched_target_indices
        ):
            continue
        atom_indices = tuple(
            sorted(
                {
                    atom
                    for reference_index in reference_indices
                    for atom in _record_atom_indices(
                        matched_by_reference[reference_index].mapped
                    )
                }
                | {
                    atom
                    for target_index in target_indices
                    for atom in _record_atom_indices(
                        target_payload.records[target_index]
                    )
                }
            )
        )
        reference_index = min(reference_indices)

        if (
            len(source_groups) != 1
            or len(target_groups) != 1
        ):
            issues.append(
                StereoValidationIssue(
                    StereoIssueCode.INVALID_DESCRIPTOR,
                    "mapped enhanced-stereo group partition does not match "
                    "the target partition",
                    record_index=reference_index,
                    atom_indices=atom_indices,
                )
            )
            continue

        source_group = source_groups[0]
        target_group = target_groups[0]
        if source_group.kind is not target_group.kind:
            issues.append(
                StereoValidationIssue(
                    StereoIssueCode.INVALID_DESCRIPTOR,
                    "mapped enhanced-stereo group kind does not match target "
                    "group kind",
                    record_index=reference_index,
                    atom_indices=atom_indices,
                )
            )
            continue

        source_partition = source_members[source_group]
        target_partition = target_members[target_group]
        mapped_target_partition = {
            match.target_index
            for match in matches
            if match.reference_index in source_partition
        }
        mapped_source_partition = {
            match.reference_index
            for match in matches
            if match.target_index in target_partition
        }
        if (
            mapped_target_partition != target_partition
            or mapped_source_partition != source_partition
        ):
            issues.append(
                StereoValidationIssue(
                    StereoIssueCode.INVALID_DESCRIPTOR,
                    "mapped enhanced-stereo group partition does not match "
                    "the target partition",
                    record_index=reference_index,
                    atom_indices=atom_indices,
                )
            )
            continue

        inversions = {
            match.transformed
            != _parse_level2_descriptor(match.target.descriptor).descriptor
            for match in component_matches
        }
        if len(inversions) > 1:
            issues.append(
                StereoValidationIssue(
                    StereoIssueCode.INVALID_DESCRIPTOR,
                    f"{source_group.kind.value} enhanced-stereo group has a "
                    "partial mapped inversion",
                    record_index=reference_index,
                    atom_indices=atom_indices,
                )
            )
    return tuple(issues)


def align_stereochemistry(
    reference: Molecule,
    target: Molecule,
    atom_mapping: Mapping[int, int],
) -> StereoAlignmentReport:
    """Compare supported descriptors under an explicit atom mapping."""

    reference_payload = get_stereo_payload(reference)
    target_payload = get_stereo_payload(target)
    mapping_issues, valid_mapping = _alignment_mapping_issues(
        reference,
        target,
        atom_mapping,
    )
    reference_projection = _gate7_projection_result(reference)
    target_projection = _gate7_projection_result(target)
    reference_validation = _validate_graph_stereochemistry(
        reference,
        projection_result=reference_projection,
    )
    target_validation = _validate_graph_stereochemistry(
        target,
        projection_result=target_projection,
    )
    issues = [*mapping_issues]
    issues.extend(_alignment_validation_issues("reference", reference_validation))
    issues.extend(_alignment_validation_issues("target", target_validation))

    target_bonds = tuple(
        tuple(bond)
        for bond in _molecule_stereo_topology(target)[2]
    )
    partially_aligned = _remap_stereo_payload(
        reference_payload,
        valid_mapping,
        allowed_bonds=target_bonds,
    )
    if issues:
        return StereoAlignmentReport(False, partially_aligned, tuple(issues))

    reference_coordination_pairs = {}
    target_coordination_pairs = {}
    for payload, projection, pairs_by_index in (
        (
            reference_payload,
            reference_projection,
            reference_coordination_pairs,
        ),
        (target_payload, target_projection, target_coordination_pairs),
    ):
        if projection is None:
            continue
        graph, _ = projection
        adjacency = _gate7_represented_adjacency(graph)
        for record_index, record in enumerate(payload.records):
            descriptor = _parse_level2_descriptor(record.descriptor).descriptor
            if descriptor.family not in _COORDINATION_FAMILIES:
                continue
            pairs_by_index[record_index] = _gate7_chelate_pairs(
                _gate7_site_from_record(record_index, record, descriptor),
                adjacency,
            )

    target_records = {}
    for target_index, record in enumerate(target_payload.records):
        parsed = _parse_level2_descriptor(record.descriptor).descriptor
        identity = _alignment_record_identity(parsed.family, record.anchor)
        target_records.setdefault(identity, []).append((target_index, record, parsed))

    aligned_records = []
    alignment_matches = []
    matched_target_indices = set()
    for reference_index, record in enumerate(reference_payload.records):
        parsed = _parse_level2_descriptor(record.descriptor).descriptor
        mapped = _mapped_alignment_record(record, valid_mapping)
        identity = _alignment_record_identity(parsed.family, mapped.anchor)
        matches = target_records.get(identity, ())
        if not matches:
            issues.append(
                StereoValidationIssue(
                    StereoIssueCode.INVALID_ANCHOR,
                    "mapped reference stereo record has no target record",
                    record_index=reference_index,
                    atom_indices=mapped.anchor.atom_indices,
                )
            )
            aligned_records.append(mapped)
            continue
        if len(matches) > 1:
            issues.append(
                StereoValidationIssue(
                    StereoIssueCode.INVALID_ANCHOR,
                    "mapped reference stereo record has duplicate target records",
                    record_index=reference_index,
                    atom_indices=mapped.anchor.atom_indices,
                )
            )
            aligned_records.append(mapped)
            continue

        target_index, target_record, _ = matches[0]
        matched_target_indices.add(target_index)
        if parsed.family in _COORDINATION_FAMILIES and (
            _gate7_mapped_chelate_pairs(
                reference_coordination_pairs[reference_index],
                valid_mapping,
            )
            != target_coordination_pairs[target_index]
        ):
            issues.append(
                StereoValidationIssue(
                    StereoIssueCode.INVALID_DESCRIPTOR,
                    "mapped coordination chelate partition does not match "
                    "target partition",
                    record_index=reference_index,
                    atom_indices=_record_atom_indices(mapped),
                )
            )
            aligned_records.append(mapped)
            continue
        transformed = _transform_supported_record(
            parsed,
            mapped,
            target_record,
        )
        if transformed.issues:
            issues.extend(
                _record_issues(
                    transformed.issues,
                    record_index=reference_index,
                    atom_indices=_record_atom_indices(mapped),
                )
            )
            aligned_records.append(mapped)
            continue

        descriptor = _render_level2_descriptor(transformed.descriptor)
        aligned = replace(
            mapped,
            anchor=target_record.anchor,
            carriers=target_record.carriers,
            descriptor=descriptor,
            group=target_record.group,
        )
        aligned_records.append(aligned)
        alignment_matches.append(
            _AlignmentMatch(
                reference_index=reference_index,
                target_index=target_index,
                mapped=aligned,
                target=target_record,
                transformed=transformed.descriptor,
            )
        )

    for target_index, record in enumerate(target_payload.records):
        if target_index not in matched_target_indices:
            issues.append(
                StereoValidationIssue(
                    StereoIssueCode.INVALID_ANCHOR,
                    "target stereo record has no mapped reference record",
                    record_index=target_index,
                    atom_indices=_record_atom_indices(record),
                )
            )

    issues.extend(
        _alignment_group_issues(
            reference_payload,
            target_payload,
            alignment_matches,
        )
    )

    aligned_payload = replace(partially_aligned, records=tuple(aligned_records))
    return StereoAlignmentReport(not issues, aligned_payload, tuple(issues))
