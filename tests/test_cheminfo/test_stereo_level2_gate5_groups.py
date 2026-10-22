from __future__ import annotations

import numpy as np
import pytest

from hotpot.cheminfo import stereo
from hotpot.cheminfo.core import Molecule


def _molecule(atom_count, bonds, *, atomic_numbers=None):
    molecule = Molecule()
    atomic_numbers = atomic_numbers or (6,) * atom_count
    for atomic_number in atomic_numbers:
        molecule.create_atom(atomic_number=atomic_number)
    for begin, end, order in bonds:
        molecule.add_bond(begin, end, order)
    return molecule


def _atom(atom):
    return stereo.StereoCarrier(stereo.StereoCarrierKind.ATOM, atom)


def _group(kind, *, scope="source", id=1):
    return stereo.StereoGroupRef(scope, kind, id)


def _tetrahedral(center, value="even", group=None):
    descriptor = f"hotpot.stereo/2/TH/{value}"
    return stereo.StereoRecord(
        anchor=stereo.StereoAnchor(stereo.StereoAnchorKind.ATOM, (center,)),
        carriers=tuple(_atom(atom) for atom in range(center + 1, center + 5)),
        descriptor=descriptor,
        group=group,
        source_namespace="hotpot.stereo/2",
        source_code=descriptor,
    )


def _axial(offset, value="even", group=None):
    descriptor = f"hotpot.stereo/2/AX/{value}"
    return stereo.StereoRecord(
        anchor=stereo.StereoAnchor(
            stereo.StereoAnchorKind.AXIS,
            (offset, offset + 1, offset + 2),
        ),
        carriers=tuple(_atom(offset + atom) for atom in (3, 4, 5, 6)),
        descriptor=descriptor,
        group=group,
        source_namespace="hotpot.stereo/2",
        source_code=descriptor,
    )


def _double_bond(group):
    descriptor = "hotpot.stereo/2/DB/together"
    return stereo.StereoRecord(
        anchor=stereo.StereoAnchor(stereo.StereoAnchorKind.BOND, (0, 1)),
        carriers=tuple(_atom(atom) for atom in (2, 3, 4, 5)),
        descriptor=descriptor,
        group=group,
        source_namespace="hotpot.stereo/2",
        source_code=descriptor,
    )


def _set_records(molecule, *records):
    payload = stereo.StereoPayload(1, tuple(records), (), ())
    stereo.set_stereo_payload(molecule, payload)
    return payload


def _multi_tetrahedral(count):
    bonds = tuple(
        (5 * block, 5 * block + carrier, 1.0)
        for block in range(count)
        for carrier in range(1, 5)
    )
    return _molecule(
        5 * count,
        bonds,
        atomic_numbers=(6, 9, 17, 35, 53) * count,
    )


def _multi_tetrahedral_coordinates(count):
    base = np.array(
        (
            (0.0, 0.0, 0.0),
            (1.0, 0.0, 0.0),
            (0.0, 1.0, 0.0),
            (0.0, 0.0, 1.0),
            (-1.0, -1.0, -1.0),
        ),
        dtype=np.float64,
    )
    return np.concatenate(
        tuple(base + (10.0 * block, 0.0, 0.0) for block in range(count))
    )


def _multi_axis(count):
    bonds = tuple(
        (7 * block + begin, 7 * block + end, order)
        for block in range(count)
        for begin, end, order in (
            (0, 1, 2.0),
            (1, 2, 2.0),
            (0, 3, 1.0),
            (0, 4, 1.0),
            (2, 5, 1.0),
            (2, 6, 1.0),
        )
    )
    return _molecule(
        7 * count,
        bonds,
        atomic_numbers=(6, 6, 6, 9, 17, 35, 53) * count,
    )


def _multi_axis_coordinates(count):
    base = np.array(
        (
            (0.0, 0.0, 0.0),
            (1.0, 0.0, 0.0),
            (2.0, 0.0, 0.0),
            (0.0, 1.0, 0.0),
            (0.0, -1.0, 0.0),
            (2.0, 0.0, 1.0),
            (2.0, 0.0, -1.0),
        ),
        dtype=np.float64,
    )
    return np.concatenate(
        tuple(base + (10.0 * block, 0.0, 0.0) for block in range(count))
    )


def _tetrahedral_and_axis_molecule():
    return _molecule(
        12,
        (
            (0, 1, 1.0),
            (0, 2, 1.0),
            (0, 3, 1.0),
            (0, 4, 1.0),
            (5, 6, 2.0),
            (6, 7, 2.0),
            (5, 8, 1.0),
            (5, 9, 1.0),
            (7, 10, 1.0),
            (7, 11, 1.0),
        ),
        atomic_numbers=(6, 9, 17, 35, 53, 6, 6, 6, 9, 17, 35, 53),
    )


def _invert_tetrahedral_geometry(coordinates, *blocks):
    inverted = coordinates.copy()
    for block in blocks:
        left, right = 5 * block + 1, 5 * block + 2
        inverted[[left, right]] = inverted[[right, left]]
    return inverted


def _invert_axis_geometry(coordinates, *blocks):
    inverted = coordinates.copy()
    for block in blocks:
        left, right = 7 * block + 3, 7 * block + 4
        inverted[[left, right]] = inverted[[right, left]]
    return inverted


def _identity_mapping(molecule):
    return {atom: atom for atom in range(len(molecule.atoms))}


@pytest.mark.parametrize(
    "kind",
    (
        stereo.StereoGroupKind.ABS,
        stereo.StereoGroupKind.AND,
        stereo.StereoGroupKind.OR,
    ),
)
def test_tetrahedral_enhanced_group_kinds_are_supported(kind):
    molecule = _multi_tetrahedral(1)
    _set_records(molecule, _tetrahedral(0, group=_group(kind)))

    assert stereo.validate_stereochemistry(molecule) == (
        stereo.StereoValidationReport(True, ())
    )


@pytest.mark.parametrize(
    "kind",
    (stereo.StereoGroupKind.AND, stereo.StereoGroupKind.OR),
)
def test_group_comparison_ignores_raw_scope_and_id(kind):
    reference = _multi_tetrahedral(2)
    target = _multi_tetrahedral(2)
    source_group = _group(kind, scope="import-a", id=4)
    target_group = _group(kind, scope="import-b", id=91)
    _set_records(
        reference,
        _tetrahedral(0, group=source_group),
        _tetrahedral(5, group=source_group),
    )
    _set_records(
        target,
        _tetrahedral(0, group=target_group),
        _tetrahedral(5, group=target_group),
    )

    report = stereo.align_stereochemistry(
        reference,
        target,
        _identity_mapping(reference),
    )

    assert report.is_equivalent
    assert report.issues == ()
    assert tuple(record.group for record in report.aligned_payload.records) == (
        target_group,
        target_group,
    )


def test_and_and_or_groups_are_chemically_distinct():
    reference = _multi_tetrahedral(2)
    target = _multi_tetrahedral(2)
    source_group = _group(stereo.StereoGroupKind.AND)
    target_group = _group(stereo.StereoGroupKind.OR)
    _set_records(
        reference,
        _tetrahedral(0, group=source_group),
        _tetrahedral(5, group=source_group),
    )
    _set_records(
        target,
        _tetrahedral(0, group=target_group),
        _tetrahedral(5, group=target_group),
    )

    report = stereo.align_stereochemistry(
        reference,
        target,
        _identity_mapping(reference),
    )

    assert not report.is_equivalent
    assert len(report.issues) == 1
    assert report.issues[0].code is stereo.StereoIssueCode.INVALID_DESCRIPTOR
    assert report.issues[0].record_index == 0
    assert report.issues[0].atom_indices == tuple(range(10))
    assert "kind" in report.issues[0].message


@pytest.mark.parametrize(
    "kind",
    (stereo.StereoGroupKind.AND, stereo.StereoGroupKind.OR),
)
def test_uniform_group_inversion_is_equivalent_and_uses_target_group_refs(kind):
    reference = _multi_tetrahedral(2)
    target = _multi_tetrahedral(2)
    source_group = _group(kind, scope="source", id=7)
    target_group = _group(kind, scope="target", id=19)
    _set_records(
        reference,
        _tetrahedral(0, "even", source_group),
        _tetrahedral(5, "even", source_group),
    )
    _set_records(
        target,
        _tetrahedral(0, "odd", target_group),
        _tetrahedral(5, "odd", target_group),
    )

    report = stereo.align_stereochemistry(
        reference,
        target,
        _identity_mapping(reference),
    )

    assert report.is_equivalent
    assert report.issues == ()
    assert tuple(record.descriptor for record in report.aligned_payload.records) == (
        "hotpot.stereo/2/TH/even",
        "hotpot.stereo/2/TH/even",
    )
    assert tuple(record.group for record in report.aligned_payload.records) == (
        target_group,
        target_group,
    )


def test_group_wide_inversion_can_span_tetrahedral_and_axial_members():
    reference = _tetrahedral_and_axis_molecule()
    target = _tetrahedral_and_axis_molecule()
    source_group = _group(stereo.StereoGroupKind.AND, scope="source", id=8)
    target_group = _group(stereo.StereoGroupKind.AND, scope="target", id=29)
    _set_records(
        reference,
        _tetrahedral(0, "even", source_group),
        _axial(5, "even", source_group),
    )
    _set_records(
        target,
        _tetrahedral(0, "odd", target_group),
        _axial(5, "odd", target_group),
    )

    report = stereo.align_stereochemistry(
        reference,
        target,
        _identity_mapping(reference),
    )

    assert report.is_equivalent
    assert report.issues == ()
    assert tuple(record.group for record in report.aligned_payload.records) == (
        target_group,
        target_group,
    )


@pytest.mark.parametrize(
    "kind",
    (stereo.StereoGroupKind.AND, stereo.StereoGroupKind.OR),
)
def test_partial_group_inversion_is_one_deterministic_group_mismatch(kind):
    reference = _multi_tetrahedral(2)
    target = _multi_tetrahedral(2)
    source_group = _group(kind, scope="source", id=2)
    target_group = _group(kind, scope="target", id=8)
    _set_records(
        reference,
        _tetrahedral(0, "even", source_group),
        _tetrahedral(5, "even", source_group),
    )
    _set_records(
        target,
        _tetrahedral(0, "odd", target_group),
        _tetrahedral(5, "even", target_group),
    )

    first = stereo.align_stereochemistry(
        reference,
        target,
        _identity_mapping(reference),
    )
    second = stereo.align_stereochemistry(
        reference,
        target,
        _identity_mapping(reference),
    )

    assert first == second
    assert not first.is_equivalent
    assert len(first.issues) == 1
    issue = first.issues[0]
    assert issue.code is stereo.StereoIssueCode.INVALID_DESCRIPTOR
    assert issue.record_index == 0
    assert issue.atom_indices == tuple(range(10))
    assert kind.value in issue.message.lower()
    assert "group" in issue.message.lower()


def test_group_partition_is_compared_by_mapped_members():
    reference = _multi_tetrahedral(3)
    target = _multi_tetrahedral(3)
    source_a = _group(stereo.StereoGroupKind.AND, scope="source", id=1)
    source_b = _group(stereo.StereoGroupKind.AND, scope="source", id=2)
    target_a = _group(stereo.StereoGroupKind.AND, scope="target", id=10)
    target_b = _group(stereo.StereoGroupKind.AND, scope="target", id=20)
    _set_records(
        reference,
        _tetrahedral(0, group=source_a),
        _tetrahedral(5, group=source_a),
        _tetrahedral(10, group=source_b),
    )
    _set_records(
        target,
        _tetrahedral(0, group=target_a),
        _tetrahedral(5, group=target_b),
        _tetrahedral(10, group=target_b),
    )

    report = stereo.align_stereochemistry(
        reference,
        target,
        _identity_mapping(reference),
    )

    assert not report.is_equivalent
    assert len(report.issues) == 1
    assert report.issues[0].record_index == 0
    assert report.issues[0].atom_indices == tuple(range(15))
    assert "partition" in report.issues[0].message


def test_group_partition_follows_nonidentity_mapping_and_target_record_order():
    reference = _multi_tetrahedral(3)
    target = _multi_tetrahedral(3)
    source_a = _group(stereo.StereoGroupKind.OR, scope="source", id=1)
    source_b = _group(stereo.StereoGroupKind.OR, scope="source", id=2)
    target_a = _group(stereo.StereoGroupKind.OR, scope="target", id=31)
    target_b = _group(stereo.StereoGroupKind.OR, scope="target", id=32)
    reference_payload = _set_records(
        reference,
        _tetrahedral(0, group=source_a),
        _tetrahedral(5, group=source_a),
        _tetrahedral(10, group=source_b),
    )
    target_payload = _set_records(
        target,
        _tetrahedral(5, group=target_b),
        _tetrahedral(10, group=target_a),
        _tetrahedral(0, group=target_a),
    )
    block_mapping = (2, 0, 1)
    mapping = {
        5 * source_block + local: 5 * target_block + local
        for source_block, target_block in enumerate(block_mapping)
        for local in range(5)
    }
    mapping_snapshot = mapping.copy()

    report = stereo.align_stereochemistry(reference, target, mapping)

    assert report.is_equivalent
    assert report.issues == ()
    assert tuple(
        record.anchor.atom_indices for record in report.aligned_payload.records
    ) == (
        (10,),
        (0,),
        (5,),
    )
    assert tuple(record.group for record in report.aligned_payload.records) == (
        target_a,
        target_a,
        target_b,
    )
    assert stereo.get_stereo_payload(reference) is reference_payload
    assert stereo.get_stereo_payload(target) is target_payload
    assert mapping == mapping_snapshot


def test_incomplete_group_does_not_hide_an_independent_fixed_mismatch():
    reference = _multi_tetrahedral(3)
    target = _multi_tetrahedral(3)
    source_group = _group(stereo.StereoGroupKind.AND, scope="source", id=1)
    target_group = _group(stereo.StereoGroupKind.AND, scope="target", id=7)
    _set_records(
        reference,
        _tetrahedral(0, "even", source_group),
        _tetrahedral(5, "even", source_group),
        _tetrahedral(10, "even"),
    )
    _set_records(
        target,
        _tetrahedral(0, "even", target_group),
        _tetrahedral(10, "odd"),
    )

    report = stereo.align_stereochemistry(
        reference,
        target,
        _identity_mapping(reference),
    )

    assert not report.is_equivalent
    assert tuple(issue.code for issue in report.issues) == (
        stereo.StereoIssueCode.INVALID_ANCHOR,
        stereo.StereoIssueCode.INVALID_DESCRIPTOR,
    )
    assert tuple(issue.record_index for issue in report.issues) == (1, 2)


@pytest.mark.parametrize("source_is_abs", (False, True))
def test_abs_and_ungrouped_records_are_equivalent_when_individually_equal(
    source_is_abs,
):
    reference = _multi_tetrahedral(1)
    target = _multi_tetrahedral(1)
    absolute = _group(stereo.StereoGroupKind.ABS, scope="absolute", id=14)
    source_group, target_group = (
        (absolute, None) if source_is_abs else (None, absolute)
    )
    _set_records(reference, _tetrahedral(0, "even", source_group))
    _set_records(target, _tetrahedral(0, "even", target_group))

    report = stereo.align_stereochemistry(
        reference,
        target,
        _identity_mapping(reference),
    )

    assert report.is_equivalent
    assert report.issues == ()
    assert report.aligned_payload.records[0].group is target_group


@pytest.mark.parametrize(
    ("source_group", "target_group"),
    (
        (None, None),
        (_group(stereo.StereoGroupKind.ABS, scope="a", id=1), None),
        (
            _group(stereo.StereoGroupKind.ABS, scope="a", id=1),
            _group(stereo.StereoGroupKind.ABS, scope="b", id=2),
        ),
    ),
)
def test_abs_and_ungrouped_records_do_not_gain_an_inversion_bit(
    source_group,
    target_group,
):
    reference = _multi_tetrahedral(1)
    target = _multi_tetrahedral(1)
    _set_records(reference, _tetrahedral(0, "even", source_group))
    _set_records(target, _tetrahedral(0, "odd", target_group))

    report = stereo.align_stereochemistry(
        reference,
        target,
        _identity_mapping(reference),
    )

    assert not report.is_equivalent
    assert len(report.issues) == 1
    assert report.issues[0].code is stereo.StereoIssueCode.INVALID_DESCRIPTOR


def test_coordinate_validation_accepts_uniform_th_group_inversion_and_assigns_it():
    molecule = _multi_tetrahedral(2)
    group = _group(stereo.StereoGroupKind.AND, scope="geometry", id=3)
    payload = _set_records(
        molecule,
        _tetrahedral(0, "even", group),
        _tetrahedral(5, "even", group),
    )
    coordinates = _invert_tetrahedral_geometry(
        _multi_tetrahedral_coordinates(2),
        0,
        1,
    )
    coordinate_snapshot = coordinates.copy()

    validation = stereo.validate_stereochemistry(molecule, coordinates=coordinates)
    assignment = stereo.assign_stereochemistry(molecule, coordinates=coordinates)

    assert validation == stereo.StereoValidationReport(True, ())
    assert assignment.unresolved == ()
    assert tuple(record.descriptor for record in assignment.payload.records) == (
        "hotpot.stereo/2/TH/odd",
        "hotpot.stereo/2/TH/odd",
    )
    assert tuple(record.group for record in assignment.payload.records) == (
        group,
        group,
    )
    assert stereo.get_stereo_payload(molecule) is payload
    np.testing.assert_array_equal(coordinates, coordinate_snapshot)


@pytest.mark.parametrize(
    "group",
    (None, _group(stereo.StereoGroupKind.ABS, scope="geometry", id=4)),
)
def test_abs_and_ungrouped_coordinate_descriptors_remain_individually_fixed(group):
    molecule = _multi_tetrahedral(1)
    _set_records(molecule, _tetrahedral(0, "even", group))
    coordinates = _invert_tetrahedral_geometry(
        _multi_tetrahedral_coordinates(1),
        0,
    )

    validation = stereo.validate_stereochemistry(molecule, coordinates=coordinates)
    assignment = stereo.assign_stereochemistry(molecule, coordinates=coordinates)

    assert not validation.is_valid
    assert validation.issues == assignment.unresolved
    assert assignment.payload.records[0].descriptor == "hotpot.stereo/2/TH/odd"
    assert len(assignment.unresolved) == 1
    assert assignment.unresolved[0].code is (
        stereo.StereoIssueCode.COORDINATE_CONTRADICTION
    )
    assert assignment.unresolved[0].record_index == 0
    assert assignment.unresolved[0].atom_indices == tuple(range(5))


@pytest.mark.parametrize(
    "kind",
    (stereo.StereoGroupKind.AND, stereo.StereoGroupKind.OR),
)
def test_partial_th_coordinate_inversion_reports_one_stable_group_issue(kind):
    molecule = _multi_tetrahedral(2)
    group = _group(kind, scope="geometry", id=6)
    _set_records(
        molecule,
        _tetrahedral(0, "even", group),
        _tetrahedral(5, "even", group),
    )
    coordinates = _invert_tetrahedral_geometry(
        _multi_tetrahedral_coordinates(2),
        0,
    )

    first = stereo.assign_stereochemistry(molecule, coordinates=coordinates)
    second = stereo.assign_stereochemistry(molecule, coordinates=coordinates)
    validation = stereo.validate_stereochemistry(molecule, coordinates=coordinates)

    assert first == second
    assert validation.issues == first.unresolved
    assert tuple(record.descriptor for record in first.payload.records) == (
        "hotpot.stereo/2/TH/odd",
        "hotpot.stereo/2/TH/even",
    )
    assert len(first.unresolved) == 1
    issue = first.unresolved[0]
    assert issue.code is stereo.StereoIssueCode.COORDINATE_CONTRADICTION
    assert issue.record_index == 0
    assert issue.atom_indices == tuple(range(10))
    assert kind.value in issue.message.lower()
    assert "group" in issue.message.lower()


def test_degenerate_group_member_does_not_create_a_partial_inversion_issue():
    molecule = _multi_tetrahedral(3)
    group = _group(stereo.StereoGroupKind.AND, scope="geometry", id=9)
    _set_records(
        molecule,
        _tetrahedral(0, "even", group),
        _tetrahedral(5, "even", group),
        _tetrahedral(10, "even", group),
    )
    coordinates = _invert_tetrahedral_geometry(
        _multi_tetrahedral_coordinates(3),
        0,
    )
    coordinates[11] = coordinates[10]

    assignment = stereo.assign_stereochemistry(molecule, coordinates=coordinates)

    assert tuple(record.descriptor for record in assignment.payload.records) == (
        "hotpot.stereo/2/TH/odd",
        "hotpot.stereo/2/TH/even",
        "hotpot.stereo/2/TH/even",
    )
    assert len(assignment.unresolved) == 1
    assert assignment.unresolved[0].code is (
        stereo.StereoIssueCode.DEGENERATE_GEOMETRY
    )
    assert assignment.unresolved[0].record_index == 2


def test_ax_group_uniform_inversion_is_equivalent_and_uses_target_group_refs():
    reference = _multi_axis(2)
    target = _multi_axis(2)
    source_group = _group(stereo.StereoGroupKind.OR, scope="source", id=3)
    target_group = _group(stereo.StereoGroupKind.OR, scope="target", id=17)
    _set_records(
        reference,
        _axial(0, "even", source_group),
        _axial(7, "even", source_group),
    )
    _set_records(
        target,
        _axial(0, "odd", target_group),
        _axial(7, "odd", target_group),
    )

    report = stereo.align_stereochemistry(
        reference,
        target,
        _identity_mapping(reference),
    )

    assert report.is_equivalent
    assert report.issues == ()
    assert tuple(record.descriptor for record in report.aligned_payload.records) == (
        "hotpot.stereo/2/AX/even",
        "hotpot.stereo/2/AX/even",
    )
    assert tuple(record.group for record in report.aligned_payload.records) == (
        target_group,
        target_group,
    )


def test_coordinate_validation_accepts_uniform_ax_group_inversion_and_assigns_it():
    molecule = _multi_axis(2)
    group = _group(stereo.StereoGroupKind.AND, scope="geometry", id=11)
    payload = _set_records(
        molecule,
        _axial(0, "even", group),
        _axial(7, "even", group),
    )
    coordinates = _invert_axis_geometry(_multi_axis_coordinates(2), 0, 1)
    coordinate_snapshot = coordinates.copy()

    validation = stereo.validate_stereochemistry(molecule, coordinates=coordinates)
    assignment = stereo.assign_stereochemistry(molecule, coordinates=coordinates)

    assert validation == stereo.StereoValidationReport(True, ())
    assert assignment.unresolved == ()
    assert tuple(record.descriptor for record in assignment.payload.records) == (
        "hotpot.stereo/2/AX/odd",
        "hotpot.stereo/2/AX/odd",
    )
    assert tuple(record.group for record in assignment.payload.records) == (
        group,
        group,
    )
    assert stereo.get_stereo_payload(molecule) is payload
    np.testing.assert_array_equal(coordinates, coordinate_snapshot)


@pytest.mark.parametrize(
    "kind",
    (
        stereo.StereoGroupKind.ABS,
        stereo.StereoGroupKind.AND,
        stereo.StereoGroupKind.OR,
    ),
)
def test_double_bond_enhanced_group_remains_explicitly_unsupported(kind):
    molecule = _molecule(
        6,
        (
            (0, 1, 2.0),
            (0, 2, 1.0),
            (0, 3, 1.0),
            (1, 4, 1.0),
            (1, 5, 1.0),
        ),
    )
    _set_records(
        molecule,
        _double_bond(_group(kind, scope="db", id=1)),
    )

    report = stereo.validate_stereochemistry(molecule)

    assert not report.is_valid
    assert len(report.issues) == 1
    issue = report.issues[0]
    assert issue.code is stereo.StereoIssueCode.UNSUPPORTED
    assert issue.record_index == 0
    assert issue.atom_indices == tuple(range(6))
    assert "DB" in issue.message
    assert "enhanced" in issue.message
