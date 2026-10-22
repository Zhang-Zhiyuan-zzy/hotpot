from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from hotpot.cheminfo import stereo
from hotpot.cheminfo.core import Molecule


def _molecule(atomic_numbers, bonds, *, implicit_hydrogens=None):
    molecule = Molecule()
    hydrogen_counts = (
        (0,) * len(atomic_numbers)
        if implicit_hydrogens is None
        else implicit_hydrogens
    )
    for atomic_number, hydrogen_count in zip(
        atomic_numbers,
        hydrogen_counts,
    ):
        molecule.create_atom(
            atomic_number=atomic_number,
            implicit_hydrogens=hydrogen_count,
        )
    for begin, end, order in bonds:
        molecule.add_bond(begin, end, order)
    return molecule


def _atom(atom):
    return stereo.StereoCarrier(stereo.StereoCarrierKind.ATOM, atom)


_H = stereo.StereoCarrier(stereo.StereoCarrierKind.IMPLICIT_H)


def _record(anchor, carriers, descriptor):
    return stereo.StereoRecord(
        anchor=anchor,
        carriers=tuple(carriers),
        descriptor=descriptor,
        group=None,
        source_namespace="hotpot.stereo/2",
        source_code=descriptor,
    )


def _tetrahedral(center, carriers, value="even"):
    return _record(
        stereo.StereoAnchor(stereo.StereoAnchorKind.ATOM, (center,)),
        carriers,
        f"hotpot.stereo/2/TH/{value}",
    )


def _double_bond(begin, end, carriers, value="together"):
    return _record(
        stereo.StereoAnchor(stereo.StereoAnchorKind.BOND, (begin, end)),
        carriers,
        f"hotpot.stereo/2/DB/{value}",
    )


def _set_records(molecule, *records):
    payload = stereo.StereoPayload(1, tuple(records), (), ())
    stereo.set_stereo_payload(molecule, payload)
    return payload


@pytest.mark.parametrize("value", ("even", "odd"))
def test_manual_tetrahedral_descriptor_validates_against_graph(value):
    molecule = _molecule(
        (14, 6, 7, 8),
        ((0, 1, 1.0), (0, 2, 1.0), (0, 3, 1.0)),
        implicit_hydrogens=(1, 0, 0, 0),
    )
    _set_records(
        molecule,
        _tetrahedral(0, (_atom(1), _atom(2), _atom(3), _H), value),
    )

    assert stereo.validate_stereochemistry(molecule) == (
        stereo.StereoValidationReport(True, ())
    )


@pytest.mark.parametrize("value", ("together", "opposite"))
def test_manual_double_bond_scopes_virtual_hydrogen_per_endpoint(value):
    molecule = _molecule(
        (6, 6, 9, 17),
        ((0, 1, 2.0), (0, 2, 1.0), (1, 3, 1.0)),
        implicit_hydrogens=(1, 1, 0, 0),
    )
    _set_records(
        molecule,
        _double_bond(0, 1, (_atom(2), _H, _atom(3), _H), value),
    )

    assert stereo.validate_stereochemistry(molecule).is_valid


def test_double_bond_scopes_shared_ring_atom_per_endpoint():
    molecule = _molecule(
        (6, 6, 7),
        ((0, 1, 2.0), (0, 2, 1.0), (1, 2, 1.0)),
        implicit_hydrogens=(1, 1, 0),
    )
    _set_records(
        molecule,
        _double_bond(0, 1, (_atom(2), _H, _atom(2), _H)),
    )

    assert stereo.validate_stereochemistry(molecule).is_valid


def test_tetrahedral_carriers_must_exactly_cover_graph_neighbours():
    molecule = _molecule(
        (14, 6, 7, 8, 9),
        ((0, 1, 1.0), (0, 2, 1.0), (0, 3, 1.0), (0, 4, 1.0)),
    )
    _set_records(
        molecule,
        _tetrahedral(0, (_atom(1), _atom(2), _atom(3), _H)),
    )

    report = stereo.validate_stereochemistry(molecule)

    assert not report.is_valid
    assert tuple(issue.code for issue in report.issues) == (
        stereo.StereoIssueCode.INVALID_CARRIER,
        stereo.StereoIssueCode.INVALID_CARRIER,
    )
    assert all(issue.record_index == 0 for issue in report.issues)
    assert report.issues[0].atom_indices == (0, 1, 2, 3, 4)


def test_tetrahedral_rejects_non_neighbour_and_non_hydrogen_virtual_carrier():
    molecule = _molecule(
        (14, 6, 7, 8, 9),
        ((0, 1, 1.0), (0, 2, 1.0), (0, 3, 1.0)),
    )
    vacancy = stereo.StereoCarrier(stereo.StereoCarrierKind.VACANCY)
    _set_records(
        molecule,
        _tetrahedral(0, (_atom(1), _atom(2), _atom(4), vacancy)),
    )

    report = stereo.validate_stereochemistry(molecule)

    assert not report.is_valid
    assert {
        issue.message for issue in report.issues
    } >= {
        "carrier atom row 4 is not adjacent to TH center 0",
        "TH permits only atom and implicit-H carriers",
    }
    assert all(issue.record_index == 0 for issue in report.issues)


def test_double_bond_requires_real_double_bond_and_endpoint_local_groups():
    molecule = _molecule(
        (6, 6, 7, 8, 9, 17),
        (
            (0, 1, 1.0),
            (0, 2, 1.0),
            (0, 3, 1.0),
            (1, 4, 1.0),
            (1, 5, 1.0),
        ),
    )
    _set_records(
        molecule,
        _double_bond(0, 1, (_atom(4), _atom(5), _atom(2), _atom(3))),
    )

    report = stereo.validate_stereochemistry(molecule)

    assert not report.is_valid
    assert report.issues[0] == stereo.StereoValidationIssue(
        stereo.StereoIssueCode.INVALID_ANCHOR,
        "DB anchor must identify a double bond",
        record_index=0,
        atom_indices=(0, 1),
    )
    assert sum(
        "not a substituent of DB endpoint" in issue.message
        for issue in report.issues
    ) == 4


@pytest.mark.parametrize(
    ("descriptor", "code"),
    (
        ("hotpot.stereo/2/TH/R", stereo.StereoIssueCode.INVALID_DESCRIPTOR),
        ("hotpot.stereo/2/NEW/value", stereo.StereoIssueCode.UNSUPPORTED),
        ("hotpot.stereo/2/SP/1", stereo.StereoIssueCode.UNSUPPORTED),
        ("hotpot.stereo/2/TBP/1", stereo.StereoIssueCode.UNSUPPORTED),
        ("hotpot.stereo/2/SPY/1", stereo.StereoIssueCode.UNSUPPORTED),
        ("hotpot.stereo/2/OH/1", stereo.StereoIssueCode.UNSUPPORTED),
    ),
)
def test_invalid_or_unsupported_descriptors_are_never_silently_accepted(
    descriptor,
    code,
):
    molecule = _molecule(
        (14, 6, 7, 8, 9),
        ((0, 1, 1.0), (0, 2, 1.0), (0, 3, 1.0), (0, 4, 1.0)),
    )
    _set_records(
        molecule,
        _record(
            stereo.StereoAnchor(stereo.StereoAnchorKind.ATOM, (0,)),
            (_atom(1), _atom(2), _atom(3), _atom(4)),
            descriptor,
        ),
    )

    report = stereo.validate_stereochemistry(molecule)

    assert not report.is_valid
    assert tuple(issue.code for issue in report.issues) == (code,)
    assert report.issues[0].record_index == 0
    assert report.issues[0].atom_indices == (0, 1, 2, 3, 4)


def test_ax_descriptor_rejects_an_atom_anchor():
    molecule = _molecule(
        (14, 6, 7, 8, 9),
        ((0, 1, 1.0), (0, 2, 1.0), (0, 3, 1.0), (0, 4, 1.0)),
    )
    _set_records(
        molecule,
        _record(
            stereo.StereoAnchor(stereo.StereoAnchorKind.ATOM, (0,)),
            (_atom(1), _atom(2), _atom(3), _atom(4)),
            "hotpot.stereo/2/AX/even",
        ),
    )

    report = stereo.validate_stereochemistry(molecule)

    assert report.issues == (
        stereo.StereoValidationIssue(
            stereo.StereoIssueCode.INVALID_ANCHOR,
            "AX requires a bond anchor with 2 atom rows or an odd axis path "
            "of at least 3 rows",
            record_index=0,
            atom_indices=(0,),
        ),
    )


def test_out_of_range_anchor_and_carrier_have_stable_context():
    molecule = _molecule((6, 7), ((0, 1, 1.0),))
    _set_records(
        molecule,
        _tetrahedral(3, (_atom(0), _atom(1), _atom(2), _H)),
    )

    report = stereo.validate_stereochemistry(molecule)

    assert report.issues == (
        stereo.StereoValidationIssue(
            stereo.StereoIssueCode.INVALID_ANCHOR,
            "anchor atom row 3 is outside molecule range 0..1",
            record_index=0,
            atom_indices=(3,),
        ),
        stereo.StereoValidationIssue(
            stereo.StereoIssueCode.INVALID_CARRIER,
            "carrier atom row 2 is outside molecule range 0..1",
            record_index=0,
            atom_indices=(2,),
        ),
    )


def test_duplicate_semantic_anchor_is_reported_on_later_record():
    molecule = _molecule(
        (14, 6, 7, 8, 9),
        ((0, 1, 1.0), (0, 2, 1.0), (0, 3, 1.0), (0, 4, 1.0)),
    )
    carriers = (_atom(1), _atom(2), _atom(3), _atom(4))
    _set_records(
        molecule,
        _tetrahedral(0, carriers, "even"),
        _tetrahedral(0, carriers, "odd"),
    )

    report = stereo.validate_stereochemistry(molecule)

    assert report.issues == (
        stereo.StereoValidationIssue(
            stereo.StereoIssueCode.INVALID_ANCHOR,
            "duplicate Level-2 stereo anchor",
            record_index=1,
            atom_indices=(0,),
        ),
    )


def test_tetrahedral_enhanced_group_is_supported_after_gate5():
    molecule = _molecule(
        (14, 6, 7, 8, 9),
        ((0, 1, 1.0), (0, 2, 1.0), (0, 3, 1.0), (0, 4, 1.0)),
    )
    record = _tetrahedral(0, (_atom(1), _atom(2), _atom(3), _atom(4)))
    _set_records(
        molecule,
        replace(
            record,
            group=stereo.StereoGroupRef(
                scope="manual-a",
                kind=stereo.StereoGroupKind.OR,
                id=1,
            ),
        ),
    )

    report = stereo.validate_stereochemistry(molecule)

    assert report == stereo.StereoValidationReport(True, ())


def test_assignment_preserves_valid_manual_payload_and_does_not_infer():
    molecule = _molecule(
        (14, 6, 7, 8),
        ((0, 1, 1.0), (0, 2, 1.0), (0, 3, 1.0)),
        implicit_hydrogens=(1, 0, 0, 0),
    )
    payload = _set_records(
        molecule,
        _tetrahedral(0, (_atom(1), _atom(2), _atom(3), _H)),
    )

    report = stereo.assign_stereochemistry(molecule)

    assert report.payload is payload
    assert report.unresolved == ()
    assert stereo.get_stereo_payload(molecule) is payload


def test_degenerate_coordinates_are_reported_after_gate4():
    molecule = _molecule(
        (14, 6, 7, 8),
        ((0, 1, 1.0), (0, 2, 1.0), (0, 3, 1.0)),
        implicit_hydrogens=(1, 0, 0, 0),
    )
    payload = _set_records(
        molecule,
        _tetrahedral(0, (_atom(1), _atom(2), _atom(3), _H)),
    )
    coordinates = np.zeros((4, 3), dtype=float)

    validation = stereo.validate_stereochemistry(
        molecule,
        coordinates=coordinates,
    )
    assignment = stereo.assign_stereochemistry(
        molecule,
        coordinates=coordinates,
    )

    assert validation.issues[-1] == stereo.StereoValidationIssue(
        stereo.StereoIssueCode.DEGENERATE_GEOMETRY,
        "TH coordinate geometry has a carrier coincident with its center "
        "within relative tolerance 1e-08",
        record_index=0,
        atom_indices=(0, 1, 2, 3),
    )
    assert assignment.payload == payload
    assert assignment.unresolved == validation.issues
    assert stereo.get_stereo_payload(molecule) is payload
