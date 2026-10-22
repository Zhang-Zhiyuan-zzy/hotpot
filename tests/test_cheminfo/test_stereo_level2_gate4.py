from __future__ import annotations

from itertools import permutations, product

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
    for atomic_number, hydrogen_count in zip(atomic_numbers, hydrogen_counts):
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


def _tetrahedral_molecule(*, implicit_h=False):
    atomic_numbers = (6, 7, 8, 9) if implicit_h else (6, 7, 8, 9, 17)
    molecule = _molecule(
        atomic_numbers,
        tuple((0, atom, 1.0) for atom in range(1, len(atomic_numbers))),
        implicit_hydrogens=(1, 0, 0, 0) if implicit_h else None,
    )
    carriers = (
        (_atom(1), _atom(2), _atom(3), _H)
        if implicit_h
        else tuple(_atom(atom) for atom in range(1, 5))
    )
    return molecule, carriers


def _tetrahedral_coordinates(*, implicit_h=False):
    points = np.array(
        (
            (0.0, 0.0, 0.0),
            (1.0, 0.0, 0.0),
            (0.0, 1.0, 0.0),
            (0.0, 0.0, 1.0),
            (-1.0, -1.0, -1.0),
        ),
        dtype=np.float64,
    )
    return points[:4] if implicit_h else points


def _alkene_molecule(*, implicit_h=False):
    if implicit_h:
        return _molecule(
            (6, 6, 9, 17),
            ((0, 1, 2.0), (0, 2, 1.0), (1, 3, 1.0)),
            implicit_hydrogens=(1, 1, 0, 0),
        ), (_atom(2), _H, _atom(3), _H)
    return _molecule(
        (6, 6, 7, 8, 9, 17),
        (
            (0, 1, 2.0),
            (0, 2, 1.0),
            (0, 3, 1.0),
            (1, 4, 1.0),
            (1, 5, 1.0),
        ),
    ), tuple(_atom(atom) for atom in (2, 3, 4, 5))


def _alkene_coordinates(*, implicit_h=False):
    points = np.array(
        (
            (0.0, 0.0, 0.0),
            (1.4, 0.0, 0.0),
            (-0.5, 1.0, 0.0),
            (-0.5, -1.0, 0.0),
            (1.9, 1.0, 0.0),
            (1.9, -1.0, 0.0),
        ),
        dtype=np.float64,
    )
    return points[[0, 1, 2, 4]] if implicit_h else points


@pytest.mark.parametrize("implicit_h", (False, True))
def test_tetrahedral_signed_volume_assigns_even_without_mutating_input(implicit_h):
    molecule, carriers = _tetrahedral_molecule(implicit_h=implicit_h)
    payload = _set_records(molecule, _tetrahedral(0, carriers, "odd"))
    coordinates = _tetrahedral_coordinates(implicit_h=implicit_h)
    coordinate_snapshot = coordinates.copy()
    molecule_snapshot = molecule.coordinates.copy()

    report = stereo.assign_stereochemistry(molecule, coordinates=coordinates)

    assert report.payload.records[0].descriptor == "hotpot.stereo/2/TH/even"
    assert report.unresolved == (
        stereo.StereoValidationIssue(
            stereo.StereoIssueCode.COORDINATE_CONTRADICTION,
            "stored descriptor hotpot.stereo/2/TH/odd contradicts coordinate "
            "descriptor hotpot.stereo/2/TH/even",
            record_index=0,
            atom_indices=tuple(range(4 if implicit_h else 5)),
        ),
    )
    assert stereo.get_stereo_payload(molecule) is payload
    np.testing.assert_array_equal(coordinates, coordinate_snapshot)
    np.testing.assert_array_equal(molecule.coordinates, molecule_snapshot)


def test_tetrahedral_reflection_and_carrier_swap_invert_parity():
    molecule, carriers = _tetrahedral_molecule()
    coordinates = _tetrahedral_coordinates()
    _set_records(molecule, _tetrahedral(0, carriers))

    reflected = coordinates.copy()
    reflected[:, 0] *= -1.0
    reflection = stereo.assign_stereochemistry(molecule, coordinates=reflected)

    _set_records(
        molecule,
        _tetrahedral(0, (carriers[1], carriers[0], *carriers[2:])),
    )
    swapped = stereo.assign_stereochemistry(molecule, coordinates=coordinates)

    assert reflection.payload.records[0].descriptor == "hotpot.stereo/2/TH/odd"
    assert swapped.payload.records[0].descriptor == "hotpot.stereo/2/TH/odd"


def test_all_tetrahedral_carrier_permutations_follow_frozen_parity():
    molecule, carriers = _tetrahedral_molecule()
    coordinates = _tetrahedral_coordinates()

    for permutation in permutations(range(4)):
        inversions = sum(
            left > right
            for index, left in enumerate(permutation)
            for right in permutation[index + 1 :]
        )
        expected = "odd" if inversions % 2 else "even"
        _set_records(
            molecule,
            _tetrahedral(0, tuple(carriers[index] for index in permutation), expected),
        )

        report = stereo.assign_stereochemistry(molecule, coordinates=coordinates)

        assert report.unresolved == ()
        assert report.payload.records[0].descriptor.endswith(f"/{expected}")


def test_tetrahedral_assignment_is_rigid_motion_and_scale_invariant():
    molecule, carriers = _tetrahedral_molecule()
    _set_records(molecule, _tetrahedral(0, carriers))
    coordinates = _tetrahedral_coordinates()
    rotation = np.array(
        ((0.0, -1.0, 0.0), (1.0, 0.0, 0.0), (0.0, 0.0, 1.0)),
        dtype=np.float64,
    )
    transformed = coordinates @ rotation.T * 1.0e7 + (8.0, -3.0, 2.0)

    baseline = stereo.assign_stereochemistry(molecule, coordinates=coordinates)
    changed = stereo.assign_stereochemistry(molecule, coordinates=transformed)

    assert baseline.payload.records == changed.payload.records
    assert baseline.unresolved == changed.unresolved == ()


def test_tetrahedral_relative_tolerance_reports_degenerate_volume():
    molecule, carriers = _tetrahedral_molecule()
    _set_records(molecule, _tetrahedral(0, carriers))
    coordinates = np.array(
        (
            (0.0, 0.0, -1.0),
            (1.0, 0.0, 0.0),
            (0.0, 1.0, 0.0),
            (1.0, 1.0, 1.0e-10),
            (0.0, 0.0, 0.0),
        ),
        dtype=np.float64,
    )

    validation = stereo.validate_stereochemistry(molecule, coordinates=coordinates)
    assignment = stereo.assign_stereochemistry(molecule, coordinates=coordinates)

    assert validation.issues == assignment.unresolved
    assert validation.issues[0].code is stereo.StereoIssueCode.DEGENERATE_GEOMETRY
    assert "relative tolerance 1e-08" in validation.issues[0].message
    assert assignment.payload == stereo.get_stereo_payload(molecule)


def test_tetrahedral_degeneracy_is_invariant_to_carrier_order():
    molecule, carriers = _tetrahedral_molecule()
    coordinates = np.array(
        (
            (0.0, 0.0, -1.0),
            (0.0, 0.0, 0.0),
            (1.0, 0.0, 0.0),
            (0.0, 1.0, 0.0),
            (1.0e4, 1.0e4, 1.0e-3),
        ),
        dtype=np.float64,
    )
    issue_sequences = []

    for ordered in (carriers, (carriers[1], carriers[2], carriers[3], carriers[0])):
        _set_records(molecule, _tetrahedral(0, ordered))
        issue_sequences.append(
            stereo.validate_stereochemistry(
                molecule,
                coordinates=coordinates,
            ).issues
        )

    assert all(
        issues[0].code is stereo.StereoIssueCode.DEGENERATE_GEOMETRY
        for issues in issue_sequences
    )
    assert all("TH signed volume" in issues[0].message for issues in issue_sequences)


def test_tetrahedral_near_coincident_carrier_uses_relative_tolerance():
    molecule, carriers = _tetrahedral_molecule()
    _set_records(molecule, _tetrahedral(0, carriers))
    coordinates = _tetrahedral_coordinates()
    coordinates[1] = (1.0e-15, 0.0, 0.0)

    report = stereo.validate_stereochemistry(molecule, coordinates=coordinates)

    assert report.issues[0].code is stereo.StereoIssueCode.DEGENERATE_GEOMETRY
    assert "carrier coincident with its center" in report.issues[0].message


def test_tetrahedral_implicit_h_with_coplanar_explicit_bonds_is_degenerate():
    molecule, carriers = _tetrahedral_molecule(implicit_h=True)
    _set_records(molecule, _tetrahedral(0, carriers))
    root_three_over_two = np.sqrt(3.0) / 2.0
    coordinates = np.array(
        (
            (0.0, 0.0, 0.0),
            (1.0, 0.0, 0.0),
            (-0.5, root_three_over_two, 0.0),
            (-0.5, -root_three_over_two, 0.0),
        ),
        dtype=np.float64,
    )

    report = stereo.validate_stereochemistry(molecule, coordinates=coordinates)

    assert report.issues[0].code is stereo.StereoIssueCode.DEGENERATE_GEOMETRY
    assert "TH signed volume" in report.issues[0].message


@pytest.mark.parametrize("implicit_h", (False, True))
def test_double_bond_transverse_relation_assigns_together(implicit_h):
    molecule, carriers = _alkene_molecule(implicit_h=implicit_h)
    _set_records(molecule, _double_bond(0, 1, carriers))

    report = stereo.assign_stereochemistry(
        molecule,
        coordinates=_alkene_coordinates(implicit_h=implicit_h),
    )

    assert report.payload.records[0].descriptor == (
        "hotpot.stereo/2/DB/together"
    )
    assert report.unresolved == ()


def test_double_bond_endpoint_swap_and_geometry_inversion_assign_opposite():
    molecule, carriers = _alkene_molecule()
    coordinates = _alkene_coordinates()
    _set_records(molecule, _double_bond(0, 1, carriers))
    inverted = coordinates.copy()
    inverted[[4, 5]] = inverted[[5, 4]]

    geometry_report = stereo.assign_stereochemistry(
        molecule,
        coordinates=inverted,
    )

    _set_records(
        molecule,
        _double_bond(
            0,
            1,
            (carriers[0], carriers[1], carriers[3], carriers[2]),
        ),
    )
    carrier_report = stereo.assign_stereochemistry(
        molecule,
        coordinates=coordinates,
    )

    assert geometry_report.payload.records[0].descriptor.endswith("/opposite")
    assert carrier_report.payload.records[0].descriptor.endswith("/opposite")


def test_double_bond_assignment_survives_anchor_and_pair_reversal():
    molecule, carriers = _alkene_molecule()
    _set_records(
        molecule,
        _double_bond(1, 0, (*carriers[2:], *carriers[:2])),
    )

    report = stereo.assign_stereochemistry(
        molecule,
        coordinates=_alkene_coordinates(),
    )

    assert report.payload.records[0].descriptor.endswith("/together")
    assert report.unresolved == ()


def test_all_double_bond_endpoint_swaps_and_anchor_orders_follow_xor_relation():
    molecule, carriers = _alkene_molecule()
    coordinates = _alkene_coordinates()

    for swap_left, swap_right, reverse_anchor in product((False, True), repeat=3):
        left = carriers[:2][::-1] if swap_left else carriers[:2]
        right = carriers[2:][::-1] if swap_right else carriers[2:]
        begin, end = (1, 0) if reverse_anchor else (0, 1)
        ordered = (*right, *left) if reverse_anchor else (*left, *right)
        expected = "opposite" if swap_left ^ swap_right else "together"
        _set_records(
            molecule,
            _double_bond(begin, end, ordered, expected),
        )

        report = stereo.assign_stereochemistry(molecule, coordinates=coordinates)

        assert report.unresolved == ()
        assert report.payload.records[0].descriptor.endswith(f"/{expected}")


def test_double_bond_orthogonal_transverse_relation_is_degenerate():
    molecule, carriers = _alkene_molecule()
    _set_records(molecule, _double_bond(0, 1, carriers))
    coordinates = _alkene_coordinates()
    coordinates[4] = (1.9, 0.0, 1.0)
    coordinates[5] = (1.9, 0.0, -1.0)

    report = stereo.validate_stereochemistry(molecule, coordinates=coordinates)

    assert report.issues[0].code is stereo.StereoIssueCode.DEGENERATE_GEOMETRY
    assert "DB transverse relation" in report.issues[0].message


def test_double_bond_near_coincident_endpoints_use_relative_tolerance():
    molecule, carriers = _alkene_molecule()
    _set_records(molecule, _double_bond(0, 1, carriers))
    coordinates = _alkene_coordinates()
    coordinates[1] = (1.0e-15, 0.0, 0.0)

    report = stereo.validate_stereochemistry(molecule, coordinates=coordinates)

    assert report.issues[0].code is stereo.StereoIssueCode.DEGENERATE_GEOMETRY
    assert "coincident bond endpoints" in report.issues[0].message


def test_double_bond_nearly_axial_carriers_use_relative_tolerance():
    molecule, carriers = _alkene_molecule()
    _set_records(molecule, _double_bond(0, 1, carriers))
    coordinates = _alkene_coordinates()
    coordinates[2] = (-1.0, 1.0e-15, 0.0)
    coordinates[3] = (-1.0, -1.0e-15, 0.0)

    report = stereo.validate_stereochemistry(molecule, coordinates=coordinates)

    assert report.issues[0].code is stereo.StereoIssueCode.DEGENERATE_GEOMETRY
    assert "degenerate transverse carriers" in report.issues[0].message


def test_coordinate_contradictions_are_record_scoped_and_deterministic():
    molecule, tetrahedral_carriers = _tetrahedral_molecule()
    alkene, alkene_carriers = _alkene_molecule()
    molecule.add_component(alkene)
    shifted_alkene_carriers = tuple(
        stereo.StereoCarrier(carrier.kind, carrier.atom + 5)
        for carrier in alkene_carriers
    )
    _set_records(
        molecule,
        _tetrahedral(0, tetrahedral_carriers, "odd"),
        _double_bond(5, 6, shifted_alkene_carriers, "opposite"),
    )
    coordinates = np.vstack(
        (_tetrahedral_coordinates(), _alkene_coordinates() + (5.0, 0.0, 0.0))
    )

    first = stereo.validate_stereochemistry(molecule, coordinates=coordinates)
    second = stereo.validate_stereochemistry(molecule, coordinates=coordinates)

    assert first == second
    assert tuple(issue.code for issue in first.issues) == (
        stereo.StereoIssueCode.COORDINATE_CONTRADICTION,
        stereo.StereoIssueCode.COORDINATE_CONTRADICTION,
    )
    assert tuple(issue.record_index for issue in first.issues) == (0, 1)


@pytest.mark.parametrize(
    ("coordinates", "message", "atom_indices"),
    (
        (np.zeros((3, 3)), "coordinate array must have shape (5, 3), got (3, 3)", ()),
        (
            np.array(
                (
                    (0.0, 0.0, 0.0),
                    (1.0, 0.0, 0.0),
                    (0.0, np.nan, 0.0),
                    (0.0, 0.0, 1.0),
                    (-1.0, -1.0, -1.0),
                )
            ),
            "coordinate array contains non-finite values",
            (2,),
        ),
    ),
)
def test_invalid_coordinate_arrays_have_stable_diagnostics(
    coordinates,
    message,
    atom_indices,
):
    molecule, carriers = _tetrahedral_molecule()
    payload = _set_records(molecule, _tetrahedral(0, carriers))

    validation = stereo.validate_stereochemistry(
        molecule,
        coordinates=coordinates,
    )
    assignment = stereo.assign_stereochemistry(
        molecule,
        coordinates=coordinates,
    )

    expected = stereo.StereoValidationIssue(
        stereo.StereoIssueCode.DEGENERATE_GEOMETRY,
        message,
        atom_indices=atom_indices,
    )
    assert validation.issues == assignment.unresolved == (expected,)
    assert assignment.payload == payload


def test_explicit_coordinates_override_molecule_coordinates():
    molecule, carriers = _tetrahedral_molecule()
    _set_records(molecule, _tetrahedral(0, carriers))
    molecule.coordinates = np.zeros((5, 3), dtype=np.float64)

    without_override = stereo.assign_stereochemistry(molecule)
    with_override = stereo.assign_stereochemistry(
        molecule,
        coordinates=_tetrahedral_coordinates(),
    )

    assert without_override.unresolved == ()
    assert with_override.unresolved == ()
    assert with_override.payload.records[0].descriptor.endswith("/even")
    np.testing.assert_array_equal(molecule.coordinates, np.zeros((5, 3)))
