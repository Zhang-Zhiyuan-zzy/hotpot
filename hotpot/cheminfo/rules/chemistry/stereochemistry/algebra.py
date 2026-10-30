"""Pure stereochemistry descriptor, permutation, record and group algebra."""

from __future__ import annotations

from collections.abc import (
    Collection,
    Sequence,
)
from dataclasses import (
    dataclass,
    replace,
)
from enum import Enum
from itertools import (
    permutations,
    product,
)
from math import factorial
from typing import Mapping
from ....structure.stereo import (
    StereoAnchor,
    StereoAnchorKind,
    StereoCarrier,
    StereoCarrierKind,
    StereoGroupKind,
    StereoGroupRef,
    StereoIssueCode,
    StereoPayload,
    StereoRecord,
    StereoValidationIssue,
)


__all__ = ()


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
class _Gate7CoordinationSite:
    """One base-valid declared coordination site behind the Gate-7 fence."""

    record_index: int
    descriptor: _Level2Descriptor
    center: int
    carrier_rows: tuple[int, ...]


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
