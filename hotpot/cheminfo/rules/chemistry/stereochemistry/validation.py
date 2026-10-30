"""Pure structural and coordinate stereochemistry report assembly."""

from __future__ import annotations

from ....structure.snapshots import (
    _Gate6Graph,
    _StereoSnapshot,
)
from ....structure.stereo import (
    StereoAnchorKind,
    StereoIssueCode,
    StereoValidationIssue,
    StereoValidationReport,
)
from .algebra import (
    _COORDINATION_FAMILIES,
    _Level2DescriptorFamily,
    _gate2_issue,
    _parse_level2_descriptor,
    _record_atom_indices,
    _record_issues,
    _validate_gate2_double_bond_carriers,
    _validate_gate5_axial_carriers,
    _validate_stereo_anchor,
    _validate_stereo_carriers,
)
from .coordinates import _gate4_coordinate_assignments
from .graph import (
    _gate7_same_center_issues,
    _snapshot_stereo_bond_kinds,
    _snapshot_stereo_topology,
    _validate_gate2_atom_ranges,
    _validate_gate2_double_bond_graph,
    _validate_gate2_tetrahedral_graph,
    _validate_gate5_axial_graph,
    _validate_gate7_coordination_semantics,
)


__all__ = ()


def _validate_gate2_stereochemistry(
    snapshot: _StereoSnapshot,
    *,
    projection_result: tuple[
        _Gate6Graph | None, tuple[StereoValidationIssue, ...]
    ]
    | None = None,
) -> StereoValidationReport:
    """Validate supported manual graph semantics through Level-2 Gate 7."""

    atom_count, adjacency, bond_orders = _snapshot_stereo_topology(snapshot)
    bond_kinds = _snapshot_stereo_bond_kinds(snapshot)
    payload = snapshot.payload
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
            atom_count,
            payload,
            descriptors,
            blocked_record_indices,
            projection_result=projection_result,
        )
    )

    return StereoValidationReport(is_valid=not issues, issues=tuple(issues))


def _coordinate_validation_report(
    snapshot: _StereoSnapshot,
    coordinates,
    graph_report: StereoValidationReport,
) -> StereoValidationReport:
    """Append coordinate diagnostics after the completed graph report."""

    if coordinates is None:
        return graph_report
    _, coordinate_issues = _gate4_coordinate_assignments(
        snapshot.payload,
        len(snapshot.atom_keys),
        coordinates,
        graph_report,
    )
    issues = (*graph_report.issues, *coordinate_issues)
    return StereoValidationReport(is_valid=not issues, issues=issues)
