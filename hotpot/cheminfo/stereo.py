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


_EMPTY_STEREO_PAYLOAD = StereoPayload(
    schema_version=1,
    records=(),
    atom_facts=(),
    bond_facts=(),
)


def _bond_key(atom_indices: Collection[int]) -> frozenset[int]:
    """Return the representation-independent key for a bond's endpoints."""

    return frozenset(atom_indices)


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
        if (
            record.anchor.kind is StereoAnchorKind.BOND
            and allowed_bond_keys is not None
            and _bond_key(anchor_indices) not in allowed_bond_keys
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

    bond_family = family in (
        _Level2DescriptorFamily.DOUBLE_BOND,
        _Level2DescriptorFamily.AXIAL,
    )
    expected_kind = (
        StereoAnchorKind.BOND if bond_family else StereoAnchorKind.ATOM
    )
    expected_count = 2 if bond_family else 1
    issues = []
    if anchor.kind is not expected_kind or len(anchor.atom_indices) != expected_count:
        issues.append(
            _algebra_issue(
                StereoIssueCode.INVALID_ANCHOR,
                f"{family.value} requires a {expected_kind.value} anchor with "
                f"{expected_count} atom row(s)",
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
    if bond_family and len(set(anchor.atom_indices)) != len(anchor.atom_indices):
        issues.append(
            _algebra_issue(
                StereoIssueCode.INVALID_ANCHOR,
                "bond anchor endpoints must be distinct",
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
        if source_anchor.kind is StereoAnchorKind.ATOM:
            same_identity = source_anchor.atom_indices == reordered_anchor.atom_indices
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
    issues.extend(_validate_stereo_carriers(source_carriers, expected_count=count))
    issues.extend(
        _validate_stereo_carriers(reordered_carriers, expected_count=count)
    )
    if issues:
        return _Level2AlgebraResult(None, tuple(issues))

    reordering = _carrier_reordering(source_carriers, reordered_carriers)
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
        anchor_reversed = (
            source_anchor.atom_indices == reordered_anchor.atom_indices[::-1]
        )
        carrier_pairs_reversed = frozenset(reordering[:2]) == frozenset((2, 3))
        if anchor_reversed != carrier_pairs_reversed:
            return _Level2AlgebraResult(
                None,
                (
                    _algebra_issue(
                        StereoIssueCode.INVALID_CARRIER,
                        "bond-anchor reversal and endpoint carrier-pair order disagree",
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
