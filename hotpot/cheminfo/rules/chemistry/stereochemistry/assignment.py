"""Pure stereochemistry proposals and bounded direct-priority names."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace
from typing import Mapping
from ....structure.stereo import (
    AtomTransportRecord,
    StereoAssignmentReport,
    StereoCarrier,
    StereoCarrierKind,
    StereoDerivedDescriptor,
    StereoGroupKind,
    StereoIssueCode,
    StereoPayload,
    StereoType,
    StereoValidationIssue,
    StereoValidationReport,
)
from .algebra import (
    _Level2DescriptorFamily,
    _gate2_issue,
    _parse_level2_descriptor,
    _record_atom_indices,
    _render_level2_descriptor,
    _transform_supported_record,
)
from .coordinates import _gate4_coordinate_assignments


__all__ = ()


def _direct_priority_order(
    atomic_numbers: Sequence[int],
    carriers: Sequence[StereoCarrier],
    atom_facts: Mapping[int, AtomTransportRecord],
) -> tuple[int, ...] | None:
    """Order carriers high-to-low when direct Z/isotope facts are complete."""

    priorities = []
    for carrier in carriers:
        if carrier.kind is StereoCarrierKind.IMPLICIT_H:
            priorities.append((1, 1))
            continue
        atomic_number = int(atomic_numbers[carrier.atom])
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
    atomic_numbers: Sequence[int],
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
                atomic_numbers,
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
                atomic_numbers,
                record.carriers[:2],
                atom_facts,
            )
            end_order = _direct_priority_order(
                atomic_numbers,
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


def _stereo_assignment_report(
    payload: StereoPayload,
    atom_count: int,
    atomic_numbers: tuple[int, ...],
    graph_report: StereoValidationReport,
    coordinates,
) -> StereoAssignmentReport:
    """Return the existing proposal, sharing unchanged transported records."""

    coordinate_issues = ()
    assigned_payload = payload
    if coordinates is not None:
        assignments, coordinate_issues = _gate4_coordinate_assignments(
            payload,
            atom_count,
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
        atomic_numbers,
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
