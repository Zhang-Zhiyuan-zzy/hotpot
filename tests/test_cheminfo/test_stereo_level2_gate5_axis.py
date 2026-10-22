from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from hotpot.cheminfo import stereo
from hotpot.cheminfo.core import Molecule


def _molecule(atom_count, bonds):
    molecule = Molecule()
    for _ in range(atom_count):
        molecule.create_atom(atomic_number=6)
    for bond in bonds:
        molecule.add_bond(*bond)
    return molecule


def _atom(atom):
    return stereo.StereoCarrier(stereo.StereoCarrierKind.ATOM, atom)


_H = stereo.StereoCarrier(stereo.StereoCarrierKind.IMPLICIT_H)


def _axial(kind, path, carriers, value="even", group=None):
    descriptor = f"hotpot.stereo/2/AX/{value}"
    return stereo.StereoRecord(
        anchor=stereo.StereoAnchor(kind, tuple(path)),
        carriers=tuple(carriers),
        descriptor=descriptor,
        group=group,
        source_namespace="hotpot.stereo/2",
        source_code=descriptor,
    )


def _set_records(molecule, *records):
    payload = stereo.StereoPayload(1, tuple(records), (), ())
    stereo.set_stereo_payload(molecule, payload)
    return payload


def _allene(*, implicit_h=False):
    if implicit_h:
        molecule = _molecule(5, ((0, 1, 2.0), (1, 2, 2.0), (0, 3, 1.0), (2, 4, 1.0)))
        carriers = (_atom(3), _H, _atom(4), _H)
    else:
        molecule = _molecule(
            7,
            (
                (0, 1, 2.0),
                (1, 2, 2.0),
                (0, 3, 1.0),
                (0, 4, 1.0),
                (2, 5, 1.0),
                (2, 6, 1.0),
            ),
        )
        carriers = tuple(_atom(atom) for atom in (3, 4, 5, 6))
    return molecule, carriers


def _allene_coordinates(*, implicit_h=False):
    coordinates = np.array(
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
    return coordinates[[0, 1, 2, 3, 5]] if implicit_h else coordinates


@pytest.mark.parametrize("implicit_h", (False, True))
def test_allene_axis_accepts_exact_endpoint_carriers(implicit_h):
    molecule, carriers = _allene(implicit_h=implicit_h)
    _set_records(
        molecule,
        _axial(stereo.StereoAnchorKind.AXIS, (0, 1, 2), carriers),
    )

    assert stereo.validate_stereochemistry(molecule) == (
        stereo.StereoValidationReport(True, ())
    )


def test_even_bond_cumulene_requires_full_maximal_odd_atom_path():
    molecule = _molecule(
        9,
        (
            (0, 1, 2.0),
            (1, 2, 2.0),
            (2, 3, 2.0),
            (3, 4, 2.0),
            (0, 5, 1.0),
            (0, 6, 1.0),
            (4, 7, 1.0),
            (4, 8, 1.0),
        ),
    )
    carriers = tuple(_atom(atom) for atom in (5, 6, 7, 8))
    _set_records(
        molecule,
        _axial(stereo.StereoAnchorKind.AXIS, range(5), carriers),
    )
    assert stereo.validate_stereochemistry(molecule).is_valid

    _set_records(
        molecule,
        _axial(stereo.StereoAnchorKind.AXIS, (0, 1, 2), carriers),
    )
    report = stereo.validate_stereochemistry(molecule)
    assert not report.is_valid
    assert any("must be maximal" in issue.message for issue in report.issues)


@pytest.mark.parametrize(
    ("path", "message"),
    (
        ((0, 1, 2, 3), "odd axis path"),
        ((0, 1, 0), "must be distinct"),
    ),
)
def test_axis_path_requires_odd_simple_shape(path, message):
    molecule, carriers = _allene()
    _set_records(
        molecule,
        _axial(stereo.StereoAnchorKind.AXIS, path, carriers),
    )

    report = stereo.validate_stereochemistry(molecule)

    assert not report.is_valid
    assert any(message in issue.message for issue in report.issues)


def test_axis_rejects_missing_nondouble_and_internal_off_axis_edges():
    molecule = _molecule(
        8,
        (
            (0, 1, 2.0),
            (1, 2, 1.0),
            (1, 7, 1.0),
            (0, 3, 1.0),
            (0, 4, 1.0),
            (2, 5, 1.0),
            (2, 6, 1.0),
        ),
    )
    _set_records(
        molecule,
        _axial(
            stereo.StereoAnchorKind.AXIS,
            (0, 1, 2),
            tuple(_atom(atom) for atom in (3, 4, 5, 6)),
        ),
    )

    messages = tuple(
        issue.message for issue in stereo.validate_stereochemistry(molecule).issues
    )
    assert "AX axis path edges must be non-aromatic double bonds" in messages
    assert "AX axis path internal atoms cannot have off-axis neighbours" in messages


@pytest.mark.parametrize("bond_kind", ("aromatic", "dative"))
def test_axis_rejects_aromatic_or_dative_double_edge(bond_kind):
    molecule, carriers = _allene()
    molecule._bonds[0]._set_bond_metadata(bond_kind=bond_kind)
    _set_records(
        molecule,
        _axial(stereo.StereoAnchorKind.AXIS, (0, 1, 2), carriers),
    )

    report = stereo.validate_stereochemistry(molecule)

    assert any("non-aromatic double" in issue.message for issue in report.issues)


@pytest.mark.parametrize("bond_kind", ("single", "unknown"))
def test_axis_requires_double_semantic_kind_even_with_numeric_order(bond_kind):
    molecule, carriers = _allene()
    molecule._bonds[0]._set_bond_metadata(bond_kind=bond_kind)
    _set_records(
        molecule,
        _axial(stereo.StereoAnchorKind.AXIS, (0, 1, 2), carriers),
    )

    report = stereo.validate_stereochemistry(molecule)

    assert any("non-aromatic double" in issue.message for issue in report.issues)


def test_formal_atrop_accepts_real_single_axis_without_barrier_claim():
    molecule = _molecule(
        6,
        ((0, 1, 1.0), (0, 2, 1.0), (0, 3, 1.0), (1, 4, 1.0), (1, 5, 1.0)),
    )
    _set_records(
        molecule,
        _axial(
            stereo.StereoAnchorKind.BOND,
            (0, 1),
            tuple(_atom(atom) for atom in (2, 3, 4, 5)),
        ),
    )

    assert stereo.validate_stereochemistry(molecule).is_valid


def test_formal_atrop_accepts_endpoint_local_implicit_hydrogens():
    molecule = _molecule(4, ((0, 1, 1.0), (0, 2, 1.0), (1, 3, 1.0)))
    _set_records(
        molecule,
        _axial(
            stereo.StereoAnchorKind.BOND,
            (0, 1),
            (_atom(2), _H, _atom(3), _H),
        ),
    )

    assert stereo.validate_stereochemistry(molecule).is_valid


def test_formal_atrop_uses_same_coordinate_and_reversal_contract():
    reference = _molecule(
        6,
        ((0, 1, 1.0), (0, 2, 1.0), (0, 3, 1.0), (1, 4, 1.0), (1, 5, 1.0)),
    )
    target = reference.copy()
    carriers = tuple(_atom(atom) for atom in (2, 3, 4, 5))
    _set_records(
        reference,
        _axial(stereo.StereoAnchorKind.BOND, (0, 1), carriers),
    )
    _set_records(
        target,
        _axial(
            stereo.StereoAnchorKind.BOND,
            (1, 0),
            (*carriers[2:], *carriers[:2]),
        ),
    )
    coordinates = np.array(
        (
            (0.0, 0.0, 0.0),
            (2.0, 0.0, 0.0),
            (0.0, 1.0, 0.0),
            (0.0, -1.0, 0.0),
            (2.0, 0.0, 1.0),
            (2.0, 0.0, -1.0),
        )
    )

    assignment = stereo.assign_stereochemistry(reference, coordinates=coordinates)
    alignment = stereo.align_stereochemistry(
        reference,
        target,
        dict(enumerate(range(6))),
    )

    assert assignment.unresolved == ()
    assert assignment.payload.records[0].descriptor.endswith("/even")
    assert alignment.is_equivalent


@pytest.mark.parametrize(
    ("order", "bond_kind"),
    ((2.0, "double"), (1.0, "aromatic"), (1.0, "dative")),
)
def test_formal_atrop_rejects_non_single_aromatic_or_dative_axis(order, bond_kind):
    molecule = _molecule(
        6,
        ((0, 1, order), (0, 2, 1.0), (0, 3, 1.0), (1, 4, 1.0), (1, 5, 1.0)),
    )
    molecule._bonds[0]._set_bond_metadata(bond_kind=bond_kind)
    _set_records(
        molecule,
        _axial(
            stereo.StereoAnchorKind.BOND,
            (0, 1),
            tuple(_atom(atom) for atom in (2, 3, 4, 5)),
        ),
    )

    assert any(
        "formal atrop AX anchor" in issue.message
        for issue in stereo.validate_stereochemistry(molecule).issues
    )


@pytest.mark.parametrize("bond_kind", ("double", "unknown"))
def test_formal_atrop_requires_single_semantic_kind_even_with_numeric_order(
    bond_kind,
):
    molecule = _molecule(
        6,
        ((0, 1, 1.0), (0, 2, 1.0), (0, 3, 1.0), (1, 4, 1.0), (1, 5, 1.0)),
    )
    molecule._bonds[0]._set_bond_metadata(bond_kind=bond_kind)
    _set_records(
        molecule,
        _axial(
            stereo.StereoAnchorKind.BOND,
            (0, 1),
            tuple(_atom(atom) for atom in (2, 3, 4, 5)),
        ),
    )

    report = stereo.validate_stereochemistry(molecule)

    assert any("formal atrop AX anchor" in issue.message for issue in report.issues)


def test_endpoint_carriers_require_exact_off_axis_coverage_and_supported_kinds():
    molecule, carriers = _allene()
    vacancy = stereo.StereoCarrier(stereo.StereoCarrierKind.VACANCY)
    _set_records(
        molecule,
        _axial(
            stereo.StereoAnchorKind.AXIS,
            (0, 1, 2),
            (carriers[0], vacancy, carriers[2], carriers[3]),
        ),
    )

    report = stereo.validate_stereochemistry(molecule)

    assert any(
        "AX permits only atom and implicit-H" in issue.message
        for issue in report.issues
    )
    assert any("exactly cover" in issue.message for issue in report.issues)


@pytest.mark.parametrize("implicit_h", (False, True))
def test_axial_coordinate_assignment_uses_frozen_triple_product(implicit_h):
    molecule, carriers = _allene(implicit_h=implicit_h)
    payload = _set_records(
        molecule,
        _axial(stereo.StereoAnchorKind.AXIS, (0, 1, 2), carriers, "odd"),
    )
    coordinates = _allene_coordinates(implicit_h=implicit_h)
    snapshot = coordinates.copy()

    report = stereo.assign_stereochemistry(molecule, coordinates=coordinates)

    assert report.payload.records[0].descriptor == "hotpot.stereo/2/AX/even"
    assert report.unresolved[0].code is stereo.StereoIssueCode.COORDINATE_CONTRADICTION
    assert stereo.get_stereo_payload(molecule) is payload
    np.testing.assert_array_equal(coordinates, snapshot)


def test_axial_coordinates_are_rigid_motion_scale_and_path_reversal_invariant():
    molecule, carriers = _allene()
    coordinates = _allene_coordinates()
    _set_records(
        molecule,
        _axial(stereo.StereoAnchorKind.AXIS, (0, 1, 2), carriers),
    )
    baseline = stereo.assign_stereochemistry(molecule, coordinates=coordinates)

    rotation = np.array(((0.0, -1.0, 0.0), (1.0, 0.0, 0.0), (0.0, 0.0, 1.0)))
    transformed = coordinates @ rotation.T * 1.0e6 + (5.0, -7.0, 2.0)
    rigid = stereo.assign_stereochemistry(molecule, coordinates=transformed)

    _set_records(
        molecule,
        _axial(
            stereo.StereoAnchorKind.AXIS,
            (2, 1, 0),
            (*carriers[2:], *carriers[:2]),
        ),
    )
    reversed_path = stereo.assign_stereochemistry(molecule, coordinates=coordinates)

    assert baseline.payload.records[0].descriptor.endswith("/even")
    assert rigid.payload.records[0].descriptor.endswith("/even")
    assert reversed_path.payload.records[0].descriptor.endswith("/even")
    assert baseline.unresolved == rigid.unresolved == reversed_path.unresolved == ()


def test_reflection_or_one_terminal_swap_inverts_axial_parity():
    molecule, carriers = _allene()
    coordinates = _allene_coordinates()
    reflected = coordinates.copy()
    reflected[:, 2] *= -1.0
    _set_records(
        molecule,
        _axial(stereo.StereoAnchorKind.AXIS, (0, 1, 2), carriers, "odd"),
    )
    reflection = stereo.assign_stereochemistry(molecule, coordinates=reflected)

    _set_records(
        molecule,
        _axial(
            stereo.StereoAnchorKind.AXIS,
            (0, 1, 2),
            (carriers[1], carriers[0], *carriers[2:]),
            "odd",
        ),
    )
    swap = stereo.assign_stereochemistry(molecule, coordinates=coordinates)

    assert reflection.payload.records[0].descriptor.endswith("/odd")
    assert swap.payload.records[0].descriptor.endswith("/odd")
    assert reflection.unresolved == swap.unresolved == ()


def test_axial_degeneracy_uses_relative_tolerance_without_linearity_cutoff():
    molecule, carriers = _allene()
    coordinates = _allene_coordinates()
    coordinates[1] = (0.8, 0.7, 0.4)
    _set_records(
        molecule,
        _axial(stereo.StereoAnchorKind.AXIS, (0, 1, 2), carriers),
    )
    assert (
        stereo.assign_stereochemistry(molecule, coordinates=coordinates).unresolved
        == ()
    )

    degenerate = coordinates.copy()
    degenerate[5:7] = ((2.0, 1.0, 1.0e-10), (2.0, -1.0, -1.0e-10))
    report = stereo.assign_stereochemistry(molecule, coordinates=degenerate)
    assert report.unresolved[0].code is stereo.StereoIssueCode.DEGENERATE_GEOMETRY


def test_axial_degeneracy_normalizes_projected_terminal_directions():
    molecule, carriers = _allene()
    coordinates = np.array(
        (
            (0.0, 0.0, 0.0),
            (0.5, 0.0, 0.0),
            (1.0, 0.0, 0.0),
            (1.0, 5.0e-5, 0.0),
            (-1.0, -5.0e-5, 0.0),
            (2.0, 0.0, 5.0e-5),
            (0.0, 0.0, -5.0e-5),
        ),
        dtype=np.float64,
    )
    _set_records(
        molecule,
        _axial(stereo.StereoAnchorKind.AXIS, (0, 1, 2), carriers),
    )

    report = stereo.assign_stereochemistry(molecule, coordinates=coordinates)

    assert report.unresolved == ()
    assert report.payload.records[0].descriptor.endswith("/even")


def test_alignment_handles_full_axis_reversal_and_endpoint_local_hydrogen():
    reference, carriers = _allene(implicit_h=True)
    target, _ = _allene(implicit_h=True)
    _set_records(
        reference,
        _axial(stereo.StereoAnchorKind.AXIS, (0, 1, 2), carriers),
    )
    _set_records(
        target,
        _axial(
            stereo.StereoAnchorKind.AXIS,
            (2, 1, 0),
            (*carriers[2:], *carriers[:2]),
        ),
    )

    report = stereo.align_stereochemistry(reference, target, dict(enumerate(range(5))))

    assert report.is_equivalent
    assert report.issues == ()


def test_alignment_terminal_swap_deterministically_transforms_parity():
    reference, carriers = _allene()
    target, _ = _allene()
    _set_records(
        reference,
        _axial(stereo.StereoAnchorKind.AXIS, (0, 1, 2), carriers, "even"),
    )
    _set_records(
        target,
        _axial(
            stereo.StereoAnchorKind.AXIS,
            (0, 1, 2),
            (carriers[1], carriers[0], *carriers[2:]),
            "odd",
        ),
    )

    report = stereo.align_stereochemistry(reference, target, dict(enumerate(range(7))))

    assert report.is_equivalent
    assert report.aligned_payload.records[0].descriptor.endswith("/odd")


def test_alignment_supports_nonidentity_axis_reversal_mapping():
    reference, carriers = _allene()
    target, _ = _allene()
    _set_records(
        reference,
        _axial(stereo.StereoAnchorKind.AXIS, (0, 1, 2), carriers),
    )
    _set_records(
        target,
        _axial(
            stereo.StereoAnchorKind.AXIS,
            (0, 1, 2),
            tuple(_atom(atom) for atom in (3, 4, 5, 6)),
        ),
    )
    mapping = {0: 2, 1: 1, 2: 0, 3: 5, 4: 6, 5: 3, 6: 4}

    report = stereo.align_stereochemistry(reference, target, mapping)

    assert report.is_equivalent
    assert report.issues == ()


def test_axial_algebra_exhausts_endpoint_swaps_and_full_path_reversal():
    descriptor = stereo._parse_level2_descriptor(
        "hotpot.stereo/2/AX/even"
    ).descriptor
    source_anchor = stereo.StereoAnchor(
        stereo.StereoAnchorKind.AXIS,
        (10, 11, 12, 13, 14),
    )
    source = tuple(_atom(atom) for atom in range(4))

    for reverse_axis in (False, True):
        for swap_begin in (False, True):
            for swap_end in (False, True):
                begin = source[:2][::-1] if swap_begin else source[:2]
                end = source[2:][::-1] if swap_end else source[2:]
                target = (*end, *begin) if reverse_axis else (*begin, *end)
                target_anchor = replace(
                    source_anchor,
                    atom_indices=(
                        source_anchor.atom_indices[::-1]
                        if reverse_axis
                        else source_anchor.atom_indices
                    ),
                )

                result = stereo._transform_level2_descriptor(
                    descriptor,
                    source,
                    target,
                    source_anchor=source_anchor,
                    reordered_anchor=target_anchor,
                )

                assert result.issues == ()
                expected = "odd" if swap_begin ^ swap_end else "even"
                assert result.descriptor.value == expected


def test_partial_axis_reordering_and_bond_axis_cross_match_are_rejected():
    descriptor = stereo._parse_level2_descriptor("hotpot.stereo/2/AX/even").descriptor
    carriers = tuple(_atom(atom) for atom in range(4))
    partial = stereo._transform_level2_descriptor(
        descriptor,
        carriers,
        carriers,
        source_anchor=stereo.StereoAnchor(
            stereo.StereoAnchorKind.AXIS,
            (4, 5, 6, 7, 8),
        ),
        reordered_anchor=stereo.StereoAnchor(
            stereo.StereoAnchorKind.AXIS,
            (4, 7, 6, 5, 8),
        ),
    )
    cross_kind = stereo._transform_gate5_axial(
        descriptor,
        _axial(stereo.StereoAnchorKind.BOND, (4, 5), carriers),
        _axial(stereo.StereoAnchorKind.AXIS, (4, 6, 5), carriers),
    )

    assert partial.issues[0].code is stereo.StereoIssueCode.INVALID_ANCHOR
    assert cross_kind.issues[0].code is stereo.StereoIssueCode.INVALID_ANCHOR


def test_enhanced_group_remains_explicitly_unsupported_at_ax_checkpoint():
    molecule, carriers = _allene()
    group = stereo.StereoGroupRef("fixture", stereo.StereoGroupKind.AND, 1)
    _set_records(
        molecule,
        _axial(stereo.StereoAnchorKind.AXIS, (0, 1, 2), carriers, group=group),
    )

    report = stereo.validate_stereochemistry(molecule)

    assert not report.is_valid
    assert report.issues[0].code is stereo.StereoIssueCode.UNSUPPORTED

