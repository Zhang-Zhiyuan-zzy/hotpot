from __future__ import annotations

from itertools import permutations

import numpy as np
import pytest

from hotpot.cheminfo import stereo
from hotpot.cheminfo.core import Molecule


def _molecule(atomic_numbers, bonds=(), *, implicit_hydrogens=None):
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
        molecule._create_atom(
            atomic_number=atomic_number,
            implicit_hydrogens=hydrogen_count,
        )
    for begin, end, order in bonds:
        molecule._add_bond(begin, end, bond_order=order)
    molecule._update_topology(clear_conformers=False)
    return molecule


def _atom(row):
    return stereo.StereoCarrier(stereo.StereoCarrierKind.ATOM, row)


_H = stereo.StereoCarrier(stereo.StereoCarrierKind.IMPLICIT_H)


def _record(
    anchor_kind,
    anchor_rows,
    carriers,
    family,
    value,
    *,
    group=None,
):
    descriptor = f"hotpot.stereo/2/{family}/{value}"
    return stereo.StereoRecord(
        anchor=stereo.StereoAnchor(anchor_kind, tuple(anchor_rows)),
        carriers=tuple(carriers),
        descriptor=descriptor,
        group=group,
        source_namespace="hotpot.stereo/2",
        source_code=descriptor,
    )


def _tetrahedral(center, carriers, value="even", *, group=None):
    return _record(
        stereo.StereoAnchorKind.ATOM,
        (center,),
        carriers,
        "TH",
        value,
        group=group,
    )


def _double_bond(begin, end, carriers, value="together"):
    return _record(
        stereo.StereoAnchorKind.BOND,
        (begin, end),
        carriers,
        "DB",
        value,
    )


def _axis(path, carriers, value="even"):
    return _record(
        stereo.StereoAnchorKind.AXIS,
        path,
        carriers,
        "AX",
        value,
    )


def _atom_fact(row, isotope):
    return stereo.AtomTransportRecord(
        atom=row,
        isotope=isotope,
        radical_electrons=None,
        current_hydrogens=None,
        source_hydrogens=None,
        source_namespace="gate6-derived-test",
    )


def _set_payload(molecule, records, *, atom_facts=()):
    payload = stereo.StereoPayload(
        schema_version=1,
        records=tuple(records),
        atom_facts=tuple(atom_facts),
        bond_facts=(),
    )
    stereo.set_stereo_payload(molecule, payload)
    return payload


def _direct_th_molecule(*, implicit_h=False):
    if implicit_h:
        molecule = _molecule(
            (14, 9, 8, 7),
            ((0, 1, 1.0), (0, 2, 1.0), (0, 3, 1.0)),
            implicit_hydrogens=(1, 0, 0, 0),
        )
        return molecule, (_atom(1), _atom(2), _atom(3), _H)
    molecule = _molecule(
        (14, 9, 8, 7, 6),
        tuple((0, row, 1.0) for row in range(1, 5)),
    )
    return molecule, tuple(_atom(row) for row in range(1, 5))


def _direct_db_molecule():
    molecule = _molecule(
        (6, 6, 9, 6, 17, 7),
        (
            (0, 1, 2.0),
            (0, 2, 1.0),
            (0, 3, 1.0),
            (1, 4, 1.0),
            (1, 5, 1.0),
        ),
    )
    return molecule, tuple(_atom(row) for row in range(2, 6))


def _derived_for(report, record_index):
    return tuple(
        item for item in report.derived if item.record_index == record_index
    )


def _priority_issue_for(report, record_index):
    return tuple(
        issue
        for issue in report.unresolved
        if (
            issue.record_index == record_index
            and issue.code is stereo.StereoIssueCode.UNSUPPORTED
        )
    )


def _assert_priority_unresolved(molecule, record_index=0):
    assert stereo.validate_stereochemistry(molecule) == (
        stereo.StereoValidationReport(True, ())
    )

    assignment = stereo.assign_stereochemistry(molecule)

    assert _derived_for(assignment, record_index) == ()
    issues = _priority_issue_for(assignment, record_index)
    assert len(issues) == 1
    return assignment, issues[0]


def _permutation_is_odd(permutation):
    return (
        sum(
            left > right
            for index, left in enumerate(permutation)
            for right in permutation[index + 1 :]
        )
        % 2
        == 1
    )


def test_derived_descriptor_is_public_and_assignment_report_is_compatible():
    descriptor = stereo.StereoDerivedDescriptor(
        record_index=3,
        stereo_type=stereo.StereoType.TETRAHEDRAL,
        label="R",
        system="CIP",
        is_absolute=True,
    )
    payload = stereo.StereoPayload(1, (), (), ())

    assert descriptor == stereo.StereoDerivedDescriptor(
        3,
        stereo.StereoType.TETRAHEDRAL,
        "R",
        "CIP",
        True,
    )
    assert "StereoDerivedDescriptor" in stereo.__all__
    assert stereo.StereoAssignmentReport(payload, ()).derived == ()


@pytest.mark.parametrize(
    ("value", "expected_label"),
    (("even", "S"), ("odd", "R")),
)
def test_th_direct_atomic_number_priority_has_frozen_rs_mapping(
    value,
    expected_label,
):
    molecule, carriers = _direct_th_molecule()
    _set_payload(molecule, (_tetrahedral(0, carriers, value),))

    report = stereo.assign_stereochemistry(molecule)

    assert report.unresolved == ()
    assert report.derived == (
        stereo.StereoDerivedDescriptor(
            0,
            stereo.StereoType.TETRAHEDRAL,
            expected_label,
            "CIP",
            True,
        ),
    )


def test_th_all_carrier_permutations_preserve_the_derived_label():
    molecule, carriers = _direct_th_molecule()

    for permutation in permutations(range(4)):
        value = "odd" if _permutation_is_odd(permutation) else "even"
        reordered = tuple(carriers[index] for index in permutation)
        _set_payload(molecule, (_tetrahedral(0, reordered, value),))

        report = stereo.assign_stereochemistry(molecule)

        assert report.unresolved == ()
        assert tuple(item.label for item in report.derived) == ("S",)


@pytest.mark.parametrize(
    ("begin", "end", "carrier_rows", "value"),
    (
        (0, 1, (2, 3, 4, 5), "together"),
        (0, 1, (3, 2, 4, 5), "opposite"),
        (0, 1, (2, 3, 5, 4), "opposite"),
        (0, 1, (3, 2, 5, 4), "together"),
        (1, 0, (4, 5, 2, 3), "together"),
        (1, 0, (5, 4, 2, 3), "opposite"),
    ),
)
def test_db_endpoint_and_anchor_permutations_preserve_z(
    begin,
    end,
    carrier_rows,
    value,
):
    molecule, _ = _direct_db_molecule()
    _set_payload(
        molecule,
        (
            _double_bond(
                begin,
                end,
                tuple(_atom(row) for row in carrier_rows),
                value,
            ),
        ),
    )

    report = stereo.assign_stereochemistry(molecule)

    assert report.unresolved == ()
    assert report.derived == (
        stereo.StereoDerivedDescriptor(
            0,
            stereo.StereoType.DOUBLE_BOND,
            "Z",
            "E/Z",
            True,
        ),
    )


def test_db_direct_priority_maps_opposite_to_e():
    molecule, carriers = _direct_db_molecule()
    _set_payload(molecule, (_double_bond(0, 1, carriers, "opposite"),))

    report = stereo.assign_stereochemistry(molecule)

    assert report.unresolved == ()
    assert tuple(item.label for item in report.derived) == ("E",)
    assert tuple(item.system for item in report.derived) == ("E/Z",)


def test_db_endpoint_and_anchor_permutations_preserve_e():
    representations = (
        (0, 1, (2, 3, 4, 5), "opposite"),
        (0, 1, (3, 2, 4, 5), "together"),
        (0, 1, (2, 3, 5, 4), "together"),
        (0, 1, (3, 2, 5, 4), "opposite"),
        (1, 0, (4, 5, 2, 3), "opposite"),
        (1, 0, (5, 4, 2, 3), "together"),
    )
    molecule, _ = _direct_db_molecule()

    for begin, end, carrier_rows, value in representations:
        _set_payload(
            molecule,
            (
                _double_bond(
                    begin,
                    end,
                    tuple(_atom(row) for row in carrier_rows),
                    value,
                ),
            ),
        )

        report = stereo.assign_stereochemistry(molecule)

        assert report.unresolved == ()
        assert tuple(item.label for item in report.derived) == ("E",)


@pytest.mark.parametrize("explicit_isotope", (None, 1))
def test_explicit_and_implicit_protium_have_the_same_direct_priority(
    explicit_isotope,
):
    implicit, implicit_carriers = _direct_th_molecule(implicit_h=True)
    _set_payload(implicit, (_tetrahedral(0, implicit_carriers),))

    explicit = _molecule(
        (14, 9, 8, 7, 1),
        tuple((0, row, 1.0) for row in range(1, 5)),
    )
    explicit_facts = (
        ()
        if explicit_isotope is None
        else (_atom_fact(4, explicit_isotope),)
    )
    _set_payload(
        explicit,
        (_tetrahedral(0, tuple(_atom(row) for row in range(1, 5))),),
        atom_facts=explicit_facts,
    )

    implicit_report = stereo.assign_stereochemistry(implicit)
    explicit_report = stereo.assign_stereochemistry(explicit)

    assert implicit_report.unresolved == explicit_report.unresolved == ()
    assert tuple(item.label for item in implicit_report.derived) == ("S",)
    assert tuple(item.label for item in explicit_report.derived) == ("S",)


def test_direct_hydrogen_isotope_priority_orders_tritium_deuterium_protium():
    molecule = _molecule(
        (6, 9, 1, 1, 1),
        tuple((0, row, 1.0) for row in range(1, 5)),
    )
    _set_payload(
        molecule,
        (_tetrahedral(0, tuple(_atom(row) for row in range(1, 5))),),
        atom_facts=(_atom_fact(2, 3), _atom_fact(3, 2)),
    )

    report = stereo.assign_stereochemistry(molecule)

    assert report.unresolved == ()
    assert tuple(item.label for item in report.derived) == ("S",)


def test_two_explicit_carbon_isotope_masses_are_directly_resolved():
    molecule = _molecule(
        (14, 9, 7, 6, 6),
        tuple((0, row, 1.0) for row in range(1, 5)),
    )
    _set_payload(
        molecule,
        (_tetrahedral(0, tuple(_atom(row) for row in range(1, 5))),),
        atom_facts=(_atom_fact(3, 13), _atom_fact(4, 12)),
    )

    report = stereo.assign_stereochemistry(molecule)

    assert report.unresolved == ()
    assert tuple(item.label for item in report.derived) == ("S",)


def test_unlabelled_carbon_versus_carbon_13_is_conservatively_unresolved():
    molecule = _molecule(
        (14, 9, 7, 6, 6),
        tuple((0, row, 1.0) for row in range(1, 5)),
    )
    _set_payload(
        molecule,
        (_tetrahedral(0, tuple(_atom(row) for row in range(1, 5))),),
        atom_facts=(_atom_fact(3, 13),),
    )

    _, issue = _assert_priority_unresolved(molecule)

    assert "direct CIP priority" in issue.message


def test_db_recursive_endpoint_tie_is_valid_but_ez_is_unresolved():
    molecule = _molecule(
        (6, 6, 6, 6, 17, 7, 9, 8),
        (
            (0, 1, 2.0),
            (0, 2, 1.0),
            (0, 3, 1.0),
            (1, 4, 1.0),
            (1, 5, 1.0),
            (2, 6, 1.0),
            (3, 7, 1.0),
        ),
    )
    _set_payload(
        molecule,
        (_double_bond(0, 1, tuple(_atom(row) for row in (2, 3, 4, 5))),),
    )

    _, issue = _assert_priority_unresolved(molecule)

    assert "direct E/Z priority" in issue.message


def _recursive_tie_molecule():
    molecule = _molecule(
        (14, 6, 6, 9, 17, 9, 8),
        (
            (0, 1, 1.0),
            (0, 2, 1.0),
            (0, 3, 1.0),
            (0, 4, 1.0),
            (1, 5, 1.0),
            (2, 6, 1.0),
        ),
    )
    _set_payload(
        molecule,
        (_tetrahedral(0, tuple(_atom(row) for row in range(1, 5))),),
    )
    return molecule


def _ring_tie_molecule():
    molecule = _molecule(
        (14, 6, 6, 9, 17, 6, 6, 7),
        (
            (0, 1, 1.0),
            (0, 2, 1.0),
            (0, 3, 1.0),
            (0, 4, 1.0),
            (1, 5, 1.0),
            (5, 6, 1.0),
            (6, 1, 1.0),
            (2, 7, 1.0),
        ),
    )
    _set_payload(
        molecule,
        (_tetrahedral(0, tuple(_atom(row) for row in range(1, 5))),),
    )
    return molecule


@pytest.mark.parametrize(
    ("case", "molecule_factory"),
    (("recursive", _recursive_tie_molecule), ("ring", _ring_tie_molecule)),
)
def test_recursive_and_ring_priorities_remain_valid_but_unassigned(
    case,
    molecule_factory,
):
    _, issue = _assert_priority_unresolved(molecule_factory())

    assert "direct CIP priority" in issue.message, case


def _descendant_stereo_tie_molecule():
    molecule = _molecule(
        (14, 6, 6, 9, 17, 7, 8, 16, 7, 8, 16),
        (
            (0, 1, 1.0),
            (0, 2, 1.0),
            (0, 3, 1.0),
            (0, 4, 1.0),
            (1, 5, 1.0),
            (1, 6, 1.0),
            (1, 7, 1.0),
            (2, 8, 1.0),
            (2, 9, 1.0),
            (2, 10, 1.0),
        ),
    )
    _set_payload(
        molecule,
        (
            _tetrahedral(0, tuple(_atom(row) for row in range(1, 5))),
            _tetrahedral(1, tuple(_atom(row) for row in (0, 5, 6, 7))),
            _tetrahedral(
                2,
                tuple(_atom(row) for row in (0, 8, 9, 10)),
                "odd",
            ),
        ),
    )
    return molecule


def test_descendant_stereo_can_distinguish_graphs_without_claiming_pseudo_cip():
    assignment, issue = _assert_priority_unresolved(
        _descendant_stereo_tie_molecule(),
        record_index=0,
    )

    assert "direct CIP priority" in issue.message
    assert all(item.record_index != 0 for item in assignment.derived)


def test_axial_cip_is_always_explicitly_outside_the_stage17_subset():
    molecule = _molecule(
        (6, 6, 6, 9, 8, 17, 7),
        (
            (0, 1, 2.0),
            (1, 2, 2.0),
            (0, 3, 1.0),
            (0, 4, 1.0),
            (2, 5, 1.0),
            (2, 6, 1.0),
        ),
    )
    _set_payload(
        molecule,
        (_axis((0, 1, 2), tuple(_atom(row) for row in range(3, 7))),),
    )

    _, issue = _assert_priority_unresolved(molecule)

    assert "axial CIP" in issue.message


@pytest.mark.parametrize(
    ("kind", "expected_absolute"),
    (
        (None, True),
        (stereo.StereoGroupKind.ABS, True),
        (stereo.StereoGroupKind.AND, False),
        (stereo.StereoGroupKind.OR, False),
    ),
)
def test_enhanced_group_kind_controls_derived_absoluteness(
    kind,
    expected_absolute,
):
    molecule, carriers = _direct_th_molecule()
    group = (
        None
        if kind is None
        else stereo.StereoGroupRef("gate6-derived", kind, 4)
    )
    _set_payload(
        molecule,
        (_tetrahedral(0, carriers, group=group),),
    )

    report = stereo.assign_stereochemistry(molecule)

    assert report.unresolved == ()
    assert tuple(item.is_absolute for item in report.derived) == (
        expected_absolute,
    )


def test_derived_descriptors_follow_stable_payload_record_order():
    molecule = _molecule(
        (14, 9, 8, 7, 6, 15, 17, 16, 14, 13),
        (
            *((0, row, 1.0) for row in range(1, 5)),
            *((5, row, 1.0) for row in range(6, 10)),
        ),
    )
    _set_payload(
        molecule,
        (
            _tetrahedral(0, tuple(_atom(row) for row in range(1, 5))),
            _tetrahedral(5, tuple(_atom(row) for row in range(6, 10)), "odd"),
        ),
    )

    report = stereo.assign_stereochemistry(molecule)

    assert report.unresolved == ()
    assert tuple(item.record_index for item in report.derived) == (0, 1)


def test_same_center_graph_issue_suppresses_every_participating_derivation():
    molecule, carriers = _direct_th_molecule()
    valid = _tetrahedral(0, carriers)
    _set_payload(molecule, (valid, valid))

    report = stereo.assign_stereochemistry(molecule)

    assert report.derived == ()
    assert tuple(issue.record_index for issue in report.unresolved) == (0, 1)
    assert all(
        issue.code is stereo.StereoIssueCode.INVALID_ANCHOR
        and issue.message
        == "atom center row 0 has multiple Level-2 geometry records"
        for issue in report.unresolved
    )


def test_global_gate6_projection_issue_suppresses_all_priority_output():
    molecule = _molecule(
        (14, 9, 8, 7, 6, 7, 29),
        tuple((0, row, 1.0) for row in range(1, 5)),
    )
    molecule._add_bond(
        5,
        6,
        bond_kind="dative",
        bond_direction="sideways",
    )
    molecule._update_topology(clear_conformers=False)
    _set_payload(
        molecule,
        (_tetrahedral(0, tuple(_atom(row) for row in range(1, 5))),),
    )

    report = stereo.assign_stereochemistry(molecule)

    assert report.derived == ()
    assert _priority_issue_for(report, 0) == ()
    assert any(
        issue.record_index is None
        and issue.code is stereo.StereoIssueCode.UNSUPPORTED
        and "dative" in issue.message
        for issue in report.unresolved
    )


def test_assignment_runs_complete_graph_validation_once(monkeypatch):
    molecule, carriers = _direct_th_molecule()
    _set_payload(molecule, (_tetrahedral(0, carriers),))
    original = stereo._validate_graph_stereochemistry
    call_count = 0

    def counted(candidate):
        nonlocal call_count
        call_count += 1
        return original(candidate)

    monkeypatch.setattr(stereo, "_validate_graph_stereochemistry", counted)

    report = stereo.assign_stereochemistry(molecule)

    assert report.unresolved == ()
    assert call_count == 1


def test_negative_hydrogen_isotope_is_not_a_direct_priority_mass():
    molecule = _molecule(
        (6, 9, 17, 1),
        ((0, 1, 1.0), (0, 2, 1.0), (0, 3, 1.0)),
        implicit_hydrogens=(1, 0, 0, 0),
    )
    _set_payload(
        molecule,
        (_tetrahedral(0, (_atom(1), _atom(2), _atom(3), _H)),),
        atom_facts=(_atom_fact(3, -1),),
    )

    _, issue = _assert_priority_unresolved(molecule)

    assert "direct CIP priority" in issue.message


def test_coordinates_derive_from_the_final_assigned_payload_without_mutation():
    molecule, carriers = _direct_th_molecule()
    payload = _set_payload(molecule, (_tetrahedral(0, carriers, "odd"),))
    coordinates = np.array(
        (
            (0.0, 0.0, 0.0),
            (1.0, 0.0, 0.0),
            (0.0, 1.0, 0.0),
            (0.0, 0.0, 1.0),
            (-1.0, -1.0, -1.0),
        ),
        dtype=np.float64,
    )
    coordinate_snapshot = coordinates.copy()

    report = stereo.assign_stereochemistry(molecule, coordinates=coordinates)

    assert report.payload.records[0].descriptor == "hotpot.stereo/2/TH/even"
    assert tuple(item.label for item in report.derived) == ("S",)
    assert any(
        issue.code is stereo.StereoIssueCode.COORDINATE_CONTRADICTION
        for issue in report.unresolved
    )
    assert stereo.get_stereo_payload(molecule) is payload
    np.testing.assert_array_equal(coordinates, coordinate_snapshot)
