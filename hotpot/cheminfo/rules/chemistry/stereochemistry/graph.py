"""Pure represented-graph projection, topology and carrier equivalence rules."""

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
from itertools import combinations
from math import isfinite
from types import MappingProxyType
from typing import Mapping
from ....structure.snapshots import (
    _Gate6Graph,
    _StereoSnapshot,
)
from ....structure.stereo import (
    AtomTransportRecord,
    StereoAnchorKind,
    StereoCarrier,
    StereoCarrierKind,
    StereoIssueCode,
    StereoPayload,
    StereoRecord,
    StereoValidationIssue,
    StereoValidationReport,
)
from .algebra import (
    _ATOM_GEOMETRY_FAMILIES,
    _AlignmentMatch,
    _COORDINATION_FAMILIES,
    _FAMILY_CARRIER_COUNTS,
    _Gate7CoordinationSite,
    _LEVEL2_DESCRIPTOR_PREFIX,
    _Level2Descriptor,
    _Level2DescriptorFamily,
    _alignment_group_issues,
    _alignment_record_identity,
    _declares_gate7_coordination,
    _gate2_issue,
    _gate7_mapped_chelate_pairs,
    _gate7_site_from_record,
    _mapped_alignment_record,
    _parse_level2_descriptor,
    _record_atom_indices,
    _record_issues,
    _transform_supported_record,
    _validate_stereo_anchor,
)


__all__ = ()

_GATE6_ROW_FIELD = "_hotpot_stereo_row"


_GATE6_H_SLOT_FIELD = "_hotpot_stereo_h_slot"


_GATE6_ATOM_FIELDS = (
    "atomic_number",
    "isotope",
    "formal_charge",
    "radical_electrons",
    "is_aromatic",
)


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
    iter_mappings,
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
    for mapping in iter_mappings(
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
    payload: StereoPayload,
    graph: _Gate6Graph,
    candidates,
    valid_record_indices: Collection[int],
    unsupported_record_indices: Collection[int],
    iter_mappings,
) -> tuple[StereoValidationIssue, ...]:
    """Prove carrier distinction with one synchronous task-local mapping callable."""

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
                iter_mappings,
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


def _snapshot_stereo_topology(
    snapshot: _StereoSnapshot,
) -> tuple[int, tuple[frozenset[int], ...], dict[frozenset[int], float]]:
    """Return atom count, adjacency, and structural bond orders by atom row."""

    atom_count = len(snapshot.atom_keys)
    adjacency = [set() for _ in range(atom_count)]
    bond_orders = {}
    atom_rows = {key: row for row, key in enumerate(snapshot.atom_keys)}
    for first_key, second_key, order, _, _ in snapshot.bonds:
        begin = atom_rows[first_key]
        end = atom_rows[second_key]
        adjacency[begin].add(end)
        adjacency[end].add(begin)
        bond_orders[frozenset((begin, end))] = float(order)
    return (
        atom_count,
        tuple(frozenset(neighbours) for neighbours in adjacency),
        bond_orders,
    )


def _snapshot_stereo_bond_kinds(snapshot: _StereoSnapshot) -> dict[frozenset[int], str]:
    """Return semantic bond kinds without invoking toolkit perception."""

    atom_rows = {key: row for row, key in enumerate(snapshot.atom_keys)}
    return {
        frozenset((atom_rows[first_key], atom_rows[second_key])): kind
        for first_key, second_key, _, kind, _ in snapshot.bonds
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


def _validate_gate7_coordination_semantics(
    atom_count: int,
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
    graph, projection_issues = projection_result
    issues = list(projection_issues)
    if graph is None:
        return tuple(issues)

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


def _freeze_gate6_target(value):
    """Freeze the exact tuple-and-dictionary native target schema recursively."""

    if isinstance(value, dict):
        return MappingProxyType({key: _freeze_gate6_target(item) for key, item in value.items()})
    if isinstance(value, tuple):
        return tuple(_freeze_gate6_target(item) for item in value)
    return value


def _gate6_preflight(snapshot: _StereoSnapshot):
    """Check only already-captured raw transport and represented-bond facts."""

    atom_count = len(snapshot.atom_keys)
    payload = snapshot.payload
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

    atom_rows = {key: row for row, key in enumerate(snapshot.atom_keys)}
    bonds_by_edge = {}
    raw_bonds = []
    for first_key, second_key, order, kind, direction in snapshot.bonds:
        first = atom_rows.get(first_key)
        second = atom_rows.get(second_key)
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
        bonds_by_edge[edge] = True
        order = float(order)
        if not isfinite(order):
            issues.append(
                _gate6_issue(
                    f"represented bond {edge} has non-finite order",
                    atom_indices=edge,
                )
            )
            continue
        directed = None
        if kind == "dative":
            if direction == "atom1_to_atom2":
                directed = (first, second)
            elif direction == "atom2_to_atom1":
                directed = (second, first)
            else:
                issues.append(
                    _gate6_issue(
                        f"dative bond {edge} has unknown direction "
                        f"{direction!r}",
                        atom_indices=edge,
                    )
                )
                continue
        raw_bonds.append((edge, order, kind, directed))

    if issues:
        return None, tuple(issues)

    adjacency_lists = [[] for _ in snapshot.atom_keys]
    for (first, second), _, _, _ in raw_bonds:
        adjacency_lists[first].append(second)
        adjacency_lists[second].append(first)
    adjacency = tuple(tuple(sorted(neighbours)) for neighbours in adjacency_lists)

    return (MappingProxyType(atom_facts), tuple(raw_bonds), adjacency), ()



def _project_gate6_graph(
    snapshot: _StereoSnapshot,
    preflight,
    prepared_hydrogens: tuple[int, ...],
    scalar_fields: tuple[tuple[int, int, bool], ...],
) -> tuple[_Gate6Graph | None, tuple[StereoValidationIssue, ...]]:
    """Project prepared immutable scalars with the existing Gate-6 chemistry."""

    atom_count = len(snapshot.atom_keys)
    atom_facts, raw_bonds, adjacency = preflight
    issues = []
    node_fields = []
    implicit_counts = []
    for row, (atomic_number, formal_charge, is_aromatic) in enumerate(scalar_fields):
        fact = atom_facts.get(row)
        isotope = 0 if fact is None or fact.isotope is None else int(fact.isotope)
        if atomic_number == 1 and isotope in (0, 1):
            isotope = 0
        radical = (
            0
            if fact is None or fact.radical_electrons is None
            else int(fact.radical_electrons)
        )
        node_fields.append(
            {
                "atomic_number": int(atomic_number),
                "isotope": isotope,
                "formal_charge": int(formal_charge),
                "radical_electrons": radical,
                "is_aromatic": bool(is_aromatic),
                _GATE6_ROW_FIELD: row,
            }
        )
        implicit_count = prepared_hydrogens[row]
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
                and scalar_fields[first][2]
                and scalar_fields[second][2]
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
    virtual_hydrogens = [[] for _ in snapshot.atom_keys]
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
            target_spec=_freeze_gate6_target({
                "keys": keys,
                "edges": tuple(edges),
                "atoms": atoms_spec,
                "bonds": tuple(bond_records),
            }),
            directed_dative_edges=frozenset(directed_dative_edges),
            real_atom_count=atom_count,
            virtual_hydrogens=tuple(
                tuple(keys) for keys in virtual_hydrogens
            ),
        ),
        (),
    )


def _gate6_stereogenicity_candidates(
    payload: StereoPayload,
    graph_report: StereoValidationReport,
):
    """Select supported records before any deferred projection or native work."""

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
    return (
        tuple(candidates),
        frozenset(valid_record_indices),
        frozenset(unsupported_record_indices),
    )
