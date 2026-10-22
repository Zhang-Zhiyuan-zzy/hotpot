"""Hotpot-owned stereochemistry transport and semantic contracts."""

from __future__ import annotations

from collections.abc import Collection, Sequence
from dataclasses import dataclass, replace
from enum import Enum
from itertools import permutations, product
from math import factorial
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


_EMPTY_STEREO_PAYLOAD = StereoPayload(
    schema_version=1,
    records=(),
    atom_facts=(),
    bond_facts=(),
)


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


def _parse_level2_descriptor(text: str) -> _Level2AlgebraResult:
    """Parse the strict, versioned Hotpot Level-2 descriptor grammar."""

    parts = text.split("/")
    if len(parts) != 4 or "/".join(parts[:2]) != _LEVEL2_DESCRIPTOR_PREFIX:
        return _Level2AlgebraResult(
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
        return _Level2AlgebraResult(
            None,
            (
                _algebra_issue(
                    StereoIssueCode.UNSUPPORTED,
                    f"unsupported Level-2 descriptor family: {parts[2]}",
                ),
            ),
        )

    value_token = parts[3]
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

    if family is _Level2DescriptorFamily.SQUARE_PYRAMIDAL:
        return _Level2AlgebraResult(
            _Level2Descriptor(family, None),
            (
                _algebra_issue(
                    StereoIssueCode.UNSUPPORTED,
                    "SPY permutation semantics are not frozen in Level-2 Gate 1",
                ),
            ),
        )

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
    if family is _Level2DescriptorFamily.SQUARE_PYRAMIDAL:
        return _Level2AlgebraResult(
            descriptor,
            (
                _algebra_issue(
                    StereoIssueCode.UNSUPPORTED,
                    "SPY permutation semantics are not frozen in Level-2 Gate 1",
                ),
            ),
        )

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
                        "ordered-anchor reversal and endpoint carrier-pair order disagree",
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

    return getattr(mol, "_stereo_payload", _EMPTY_STEREO_PAYLOAD)


def set_stereo_payload(mol: Molecule, payload: StereoPayload) -> None:
    """Store Level-1 facts without chemical interpretation."""

    mol._stereo_payload = payload


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


def _validate_gate2_stereochemistry(mol: Molecule) -> StereoValidationReport:
    """Validate supported manual graph semantics through Level-2 Gate 5."""

    atom_count, adjacency, bond_orders = _molecule_stereo_topology(mol)
    bond_kinds = _molecule_stereo_bond_kinds(mol)
    issues = []
    seen_anchors = set()
    for record_index, record in enumerate(get_stereo_payload(mol).records):
        record_atoms = _record_atom_indices(record)
        parsed = _parse_level2_descriptor(record.descriptor)
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

        family = parsed.descriptor.family
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

        if record.group is not None:
            issues.append(
                _gate2_issue(
                    StereoIssueCode.UNSUPPORTED,
                    "enhanced-stereo group semantics are deferred to "
                    "Level-2 Gate 5",
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
            anchor_identity = (
                family,
                record.anchor.atom_indices[0]
                if family is _Level2DescriptorFamily.TETRAHEDRAL
                else (
                    frozenset(record.anchor.atom_indices)
                    if record.anchor.kind is StereoAnchorKind.BOND
                    else min(
                        record.anchor.atom_indices,
                        record.anchor.atom_indices[::-1],
                    )
                ),
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
        else:
            assigned, geometry_issue = _axial_coordinate_descriptor(
                record,
                array,
                record_index=record_index,
            )
        if geometry_issue is not None:
            issues.append(geometry_issue)
            continue
        assignments[record_index] = assigned
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
    return assignments, tuple(issues)


def validate_stereochemistry(
    mol: Molecule,
    *,
    coordinates: NDArray[np.float64] | None = None,
) -> StereoValidationReport:
    """Validate Gate-2 graph and optional Gate-4 coordinate semantics."""

    graph_report = _validate_gate2_stereochemistry(mol)
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
    """Assign supported descriptors from explicit coordinates without mutation."""

    payload = get_stereo_payload(mol)
    graph_report = _validate_gate2_stereochemistry(mol)
    if coordinates is None:
        return StereoAssignmentReport(
            payload=payload,
            unresolved=graph_report.issues,
        )
    assignments, coordinate_issues = _gate4_coordinate_assignments(
        mol,
        coordinates,
        graph_report,
    )
    records = tuple(
        replace(record, descriptor=_render_level2_descriptor(assignments[index]))
        if index in assignments
        else record
        for index, record in enumerate(payload.records)
    )
    return StereoAssignmentReport(
        payload=replace(payload, records=records),
        unresolved=(*graph_report.issues, *coordinate_issues),
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

    if family is _Level2DescriptorFamily.TETRAHEDRAL:
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
    reference_validation = validate_stereochemistry(reference)
    target_validation = validate_stereochemistry(target)
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

    target_records = {}
    for target_index, record in enumerate(target_payload.records):
        parsed = _parse_level2_descriptor(record.descriptor).descriptor
        identity = _alignment_record_identity(parsed.family, record.anchor)
        target_records.setdefault(identity, []).append((target_index, record, parsed))

    aligned_records = []
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
        if parsed.family is _Level2DescriptorFamily.TETRAHEDRAL:
            transformed = _transform_level2_descriptor(
                parsed,
                mapped.carriers,
                target_record.carriers,
                source_anchor=mapped.anchor,
                reordered_anchor=target_record.anchor,
            )
        elif parsed.family is _Level2DescriptorFamily.DOUBLE_BOND:
            transformed = _transform_gate3_double_bond(
                parsed,
                mapped,
                target_record,
            )
        else:
            transformed = _transform_gate5_axial(
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
        )
        aligned_records.append(aligned)
        if descriptor != target_record.descriptor:
            issues.append(
                StereoValidationIssue(
                    StereoIssueCode.INVALID_DESCRIPTOR,
                    f"mapped descriptor {descriptor} does not match target "
                    f"descriptor {target_record.descriptor}",
                    record_index=reference_index,
                    atom_indices=_record_atom_indices(aligned),
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

    aligned_payload = replace(partially_aligned, records=tuple(aligned_records))
    return StereoAlignmentReport(not issues, aligned_payload, tuple(issues))
