"""Pure stereochemistry mapping diagnostics and alignment reports."""

from __future__ import annotations

from dataclasses import replace
from typing import Mapping
from ....structure.snapshots import _StereoSnapshot
from ....structure.stereo import (
    StereoAlignmentReport,
    StereoIssueCode,
    StereoPayload,
    StereoValidationIssue,
    StereoValidationReport,
)
from .algebra import (
    _AlignmentMatch,
    _COORDINATION_FAMILIES,
    _alignment_group_issues,
    _alignment_record_identity,
    _gate7_mapped_chelate_pairs,
    _gate7_site_from_record,
    _mapped_alignment_record,
    _parse_level2_descriptor,
    _record_atom_indices,
    _record_issues,
    _remap_stereo_payload,
    _render_level2_descriptor,
    _transform_supported_record,
)
from .graph import (
    _gate7_chelate_pairs,
    _gate7_represented_adjacency,
    _snapshot_stereo_topology,
)


__all__ = ()


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
    reference_count: int,
    target_count: int,
    reference_payload: StereoPayload,
    target_payload: StereoPayload,
    mapping_entries: tuple[tuple[object, object], ...],
) -> tuple[tuple[StereoValidationIssue, ...], dict[int, int]]:
    """Validate a one-to-one mapping over all stereo-relevant atom rows."""

    issues = []
    valid_mapping = {}
    for source, destination in mapping_entries:
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
            for record in reference_payload.records
            for atom in _record_atom_indices(record)
        }
    )
    target_atoms = sorted(
        {
            atom
            for record in target_payload.records
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


def _align_stereo_payloads(
    reference: _StereoSnapshot,
    target: _StereoSnapshot,
    valid_mapping: Mapping[int, int],
    mapping_issues: tuple[StereoValidationIssue, ...],
    reference_validation: StereoValidationReport,
    target_validation: StereoValidationReport,
    reference_projection,
    target_projection,
) -> StereoAlignmentReport:
    """Assemble the original ordered alignment report from captured inputs."""

    reference_payload = reference.payload
    target_payload = target.payload
    issues = [*mapping_issues]
    issues.extend(_alignment_validation_issues("reference", reference_validation))
    issues.extend(_alignment_validation_issues("target", target_validation))

    target_bonds = tuple(
        tuple(bond)
        for bond in _snapshot_stereo_topology(target)[2]
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
