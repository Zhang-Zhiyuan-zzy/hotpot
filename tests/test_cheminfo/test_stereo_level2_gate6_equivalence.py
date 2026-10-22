from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
from itertools import combinations

import pytest

from hotpot.cheminfo import stereo
from hotpot.cheminfo.core import Molecule
from hotpot.cheminfo.search.native import _NativeSpecTarget


def _molecule(
    atomic_numbers,
    bonds=(),
    *,
    implicit_hydrogens=None,
):
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
    molecule._update_graph(clear_conformers=False)
    return molecule


def _atom(row):
    return stereo.StereoCarrier(stereo.StereoCarrierKind.ATOM, row)


_H = stereo.StereoCarrier(stereo.StereoCarrierKind.IMPLICIT_H)


def _record(anchor_kind, anchor_rows, carriers, family, value):
    descriptor = f"hotpot.stereo/2/{family}/{value}"
    return stereo.StereoRecord(
        anchor=stereo.StereoAnchor(anchor_kind, tuple(anchor_rows)),
        carriers=tuple(carriers),
        descriptor=descriptor,
        group=None,
        source_namespace="hotpot.stereo/2",
        source_code=descriptor,
    )


def _tetrahedral(center, carriers, value="even"):
    return _record(
        stereo.StereoAnchorKind.ATOM,
        (center,),
        carriers,
        "TH",
        value,
    )


def _double_bond(begin, end, carriers, value="together"):
    return _record(
        stereo.StereoAnchorKind.BOND,
        (begin, end),
        carriers,
        "DB",
        value,
    )


def _axis(kind, path, carriers, value="even"):
    return _record(kind, path, carriers, "AX", value)


def _unsupported_sp(center):
    return _record(
        stereo.StereoAnchorKind.ATOM,
        (center,),
        (),
        "SP",
        "1",
    )


def _atom_fact(row, *, isotope=None, radical=None):
    return stereo.AtomTransportRecord(
        atom=row,
        isotope=isotope,
        radical_electrons=radical,
        current_hydrogens=None,
        source_hydrogens=None,
        source_namespace="gate6-test",
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


def _issues_for(report, record_index, code):
    return tuple(
        issue
        for issue in report.issues
        if issue.record_index == record_index and issue.code is code
    )


def _assert_non_stereogenic(molecule, record_index=0):
    report = stereo.validate_stereochemistry(molecule)

    assert not report.is_valid
    assert len(
        _issues_for(
            report,
            record_index,
            stereo.StereoIssueCode.NON_STEREOGENIC,
        )
    ) == 1
    return report


@pytest.mark.parametrize("equivalent_pair", tuple(combinations(range(4), 2)))
def test_th_is_non_stereogenic_when_any_carrier_pair_is_equivalent(
    equivalent_pair,
):
    carrier_elements = [6, 7, 8, 9]
    first, second = equivalent_pair
    carrier_elements[second] = carrier_elements[first]
    molecule = _molecule(
        (14, *carrier_elements),
        tuple((0, row, 1.0) for row in range(1, 5)),
    )
    _set_payload(
        molecule,
        (_tetrahedral(0, tuple(_atom(row) for row in range(1, 5))),),
    )

    _assert_non_stereogenic(molecule)


@pytest.mark.parametrize("equivalent_endpoint", ("begin", "end"))
def test_db_is_non_stereogenic_when_either_endpoint_pair_is_equivalent(
    equivalent_endpoint,
):
    carrier_elements = [6, 7, 8, 9]
    offset = 0 if equivalent_endpoint == "begin" else 2
    carrier_elements[offset + 1] = carrier_elements[offset]
    molecule = _molecule(
        (6, 6, *carrier_elements),
        (
            (0, 1, 2.0),
            (0, 2, 1.0),
            (0, 3, 1.0),
            (1, 4, 1.0),
            (1, 5, 1.0),
        ),
    )
    _set_payload(
        molecule,
        (_double_bond(0, 1, tuple(_atom(row) for row in range(2, 6))),),
    )

    _assert_non_stereogenic(molecule)


@pytest.mark.parametrize("equivalent_endpoint", ("begin", "end"))
def test_ax_is_non_stereogenic_when_either_endpoint_pair_is_equivalent(
    equivalent_endpoint,
):
    carrier_elements = [6, 7, 8, 9]
    offset = 0 if equivalent_endpoint == "begin" else 2
    carrier_elements[offset + 1] = carrier_elements[offset]
    molecule = _molecule(
        (6, 6, 6, *carrier_elements),
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
        (
            _axis(
                stereo.StereoAnchorKind.AXIS,
                (0, 1, 2),
                tuple(_atom(row) for row in range(3, 7)),
            ),
        ),
    )

    _assert_non_stereogenic(molecule)


@pytest.mark.parametrize(
    "distinguishing_fact",
    ("element", "isotope", "formal_charge", "radical"),
)
def test_exact_atom_labels_distinguish_otherwise_equal_th_carriers(
    distinguishing_fact,
):
    carrier_elements = [6, 6, 9, 17]
    atom_facts = ()
    if distinguishing_fact == "element":
        carrier_elements[1] = 7
    elif distinguishing_fact == "isotope":
        atom_facts = (
            _atom_fact(1, isotope=12),
            _atom_fact(2, isotope=13),
        )
    elif distinguishing_fact == "formal_charge":
        pass
    else:
        atom_facts = (
            _atom_fact(1, radical=0),
            _atom_fact(2, radical=1),
        )

    molecule = _molecule(
        (14, *carrier_elements),
        tuple((0, row, 1.0) for row in range(1, 5)),
    )
    if distinguishing_fact == "formal_charge":
        molecule.atoms[2].formal_charge = 1
    _set_payload(
        molecule,
        (_tetrahedral(0, tuple(_atom(row) for row in range(1, 5))),),
        atom_facts=atom_facts,
    )

    assert stereo.validate_stereochemistry(molecule) == (
        stereo.StereoValidationReport(True, ())
    )


@pytest.mark.parametrize("explicit_isotope", (None, 1))
def test_ordinary_explicit_hydrogen_equals_implicit_protium(explicit_isotope):
    molecule = _molecule(
        (6, 1, 9, 17),
        ((0, 1, 1.0), (0, 2, 1.0), (0, 3, 1.0)),
        implicit_hydrogens=(1, 0, 0, 0),
    )
    _set_payload(
        molecule,
        (_tetrahedral(0, (_atom(1), _H, _atom(2), _atom(3))),),
        atom_facts=(_atom_fact(1, isotope=explicit_isotope),),
    )

    _assert_non_stereogenic(molecule)


@pytest.mark.parametrize("explicit_isotope", (2, 3))
def test_explicit_deuterium_or_tritium_differs_from_implicit_protium(
    explicit_isotope,
):
    molecule = _molecule(
        (6, 1, 9, 17),
        ((0, 1, 1.0), (0, 2, 1.0), (0, 3, 1.0)),
        implicit_hydrogens=(1, 0, 0, 0),
    )
    _set_payload(
        molecule,
        (_tetrahedral(0, (_atom(1), _H, _atom(2), _atom(3))),),
        atom_facts=(_atom_fact(1, isotope=explicit_isotope),),
    )

    assert stereo.validate_stereochemistry(molecule) == (
        stereo.StereoValidationReport(True, ())
    )


@pytest.mark.parametrize("implicit_count", (0, 2))
def test_implicit_carrier_requires_one_represented_virtual_hydrogen(
    implicit_count,
):
    molecule = _molecule(
        (14, 6, 7, 8),
        ((0, 1, 1.0), (0, 2, 1.0), (0, 3, 1.0)),
        implicit_hydrogens=(implicit_count, 0, 0, 0),
    )
    _set_payload(
        molecule,
        (_tetrahedral(0, (_atom(1), _atom(2), _atom(3), _H)),),
    )

    report = stereo.validate_stereochemistry(molecule)

    assert _issues_for(
        report,
        0,
        stereo.StereoIssueCode.UNSUPPORTED,
    )
    assert not _issues_for(
        report,
        0,
        stereo.StereoIssueCode.NON_STEREOGENIC,
    )


def _hidden_equivalence_molecule(*, hide_second_oxygen):
    molecule = _molecule(
        (14, 6, 6, 9, 17, 8, 8),
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
    if hide_second_oxygen:
        molecule.hide_bonds(molecule.bonds[-1], clear_conformers=False)
    return molecule


def test_active_and_hidden_represented_bonds_give_the_same_equivalence_result():
    active = _hidden_equivalence_molecule(hide_second_oxygen=False)
    hidden = _hidden_equivalence_molecule(hide_second_oxygen=True)

    active_report = _assert_non_stereogenic(active)
    hidden_report = _assert_non_stereogenic(hidden)

    assert hidden_report == active_report


def test_aromatic_kekule_bonds_normalize_but_biaryl_axis_remains_single():
    molecule = _molecule(
        (6,) * 12,
        (
            (0, 1, 1.0),
            (0, 2, 1.0),
            (2, 3, 2.0),
            (3, 4, 1.0),
            (4, 5, 2.0),
            (5, 6, 1.0),
            (6, 0, 2.0),
            (1, 7, 2.0),
            (7, 8, 1.0),
            (8, 9, 2.0),
            (9, 10, 1.0),
            (10, 11, 2.0),
            (11, 1, 1.0),
        ),
    )
    for atom in molecule.atoms:
        atom.is_aromatic = True
    _set_payload(
        molecule,
        (
            _axis(
                stereo.StereoAnchorKind.BOND,
                (0, 1),
                tuple(_atom(row) for row in (2, 6, 7, 11)),
            ),
        ),
    )

    report = _assert_non_stereogenic(molecule)

    assert not _issues_for(
        report,
        0,
        stereo.StereoIssueCode.INVALID_ANCHOR,
    )


def _dative_direction_molecule(*, unknown_direction=False):
    molecule = _molecule(
        (14, 6, 6, 9, 17, 29, 29),
        (
            (0, 1, 1.0),
            (0, 2, 1.0),
            (0, 3, 1.0),
            (0, 4, 1.0),
        ),
    )
    first_direction = "sideways" if unknown_direction else "atom1_to_atom2"
    molecule._add_bond(
        1,
        5,
        bond_order=1.0,
        bond_kind="dative",
        bond_direction=first_direction,
    )
    molecule._add_bond(
        2,
        6,
        bond_order=1.0,
        bond_kind="dative",
        bond_direction="atom2_to_atom1",
    )
    molecule._update_graph(clear_conformers=False)
    _set_payload(
        molecule,
        (_tetrahedral(0, tuple(_atom(row) for row in range(1, 5))),),
    )
    return molecule


def test_dative_direction_prevents_a_false_substituent_automorphism():
    molecule = _dative_direction_molecule()

    assert stereo.validate_stereochemistry(molecule) == (
        stereo.StereoValidationReport(True, ())
    )


def test_unknown_dative_direction_is_publicly_unsupported():
    molecule = _dative_direction_molecule(unknown_direction=True)

    report = stereo.validate_stereochemistry(molecule)

    assert not report.is_valid
    assert any(
        issue.code is stereo.StereoIssueCode.UNSUPPORTED
        and "dative" in issue.message
        and "direction" in issue.message
        for issue in report.issues
    )


def _descendant_stereo_molecule(family, *, right_value):
    if family == "TH":
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
        descendants = (
            _tetrahedral(1, tuple(_atom(row) for row in (0, 5, 6, 7))),
            _tetrahedral(
                2,
                tuple(_atom(row) for row in (0, 8, 9, 10)),
                right_value,
            ),
        )
    elif family == "DB":
        molecule = _molecule(
            (14, 6, 6, 9, 17, 6, 7, 8, 16, 6, 7, 8, 16),
            (
                (0, 1, 1.0),
                (0, 2, 1.0),
                (0, 3, 1.0),
                (0, 4, 1.0),
                (1, 5, 2.0),
                (1, 6, 1.0),
                (5, 7, 1.0),
                (5, 8, 1.0),
                (2, 9, 2.0),
                (2, 10, 1.0),
                (9, 11, 1.0),
                (9, 12, 1.0),
            ),
        )
        descendants = (
            _double_bond(
                1,
                5,
                tuple(_atom(row) for row in (0, 6, 7, 8)),
            ),
            _double_bond(
                2,
                9,
                tuple(_atom(row) for row in (0, 10, 11, 12)),
                right_value,
            ),
        )
    else:
        molecule = _molecule(
            (14, 6, 6, 9, 17, 6, 6, 7, 8, 16, 6, 6, 7, 8, 16),
            (
                (0, 1, 1.0),
                (0, 2, 1.0),
                (0, 3, 1.0),
                (0, 4, 1.0),
                (1, 5, 2.0),
                (5, 6, 2.0),
                (1, 7, 1.0),
                (6, 8, 1.0),
                (6, 9, 1.0),
                (2, 10, 2.0),
                (10, 11, 2.0),
                (2, 12, 1.0),
                (11, 13, 1.0),
                (11, 14, 1.0),
            ),
        )
        descendants = (
            _axis(
                stereo.StereoAnchorKind.AXIS,
                (1, 5, 6),
                tuple(_atom(row) for row in (0, 7, 8, 9)),
            ),
            _axis(
                stereo.StereoAnchorKind.AXIS,
                (2, 10, 11),
                tuple(_atom(row) for row in (0, 12, 13, 14)),
                right_value,
            ),
        )

    parent = _tetrahedral(0, tuple(_atom(row) for row in range(1, 5)))
    _set_payload(molecule, (parent, *descendants))
    return molecule


@pytest.mark.parametrize(
    ("family", "same_value", "opposite_value"),
    (
        ("TH", "even", "odd"),
        ("DB", "together", "opposite"),
        ("AX", "even", "odd"),
    ),
)
def test_supported_descendant_stereo_is_preserved_by_equivalence(
    family,
    same_value,
    opposite_value,
):
    same = _descendant_stereo_molecule(family, right_value=same_value)
    opposite = _descendant_stereo_molecule(family, right_value=opposite_value)

    assert stereo.validate_stereochemistry(opposite) == (
        stereo.StereoValidationReport(True, ())
    )
    _assert_non_stereogenic(same, record_index=0)


@pytest.mark.parametrize(
    "kind",
    (stereo.StereoGroupKind.AND, stereo.StereoGroupKind.OR),
)
def test_uniform_descendant_group_inversion_preserves_equivalence(kind):
    molecule = _descendant_stereo_molecule("TH", right_value="odd")
    payload = stereo.get_stereo_payload(molecule)
    group = stereo.StereoGroupRef("gate6-group", kind, 7)
    stereo.set_stereo_payload(
        molecule,
        replace(
            payload,
            records=(
                payload.records[0],
                replace(payload.records[1], group=group),
                replace(payload.records[2], group=group),
            ),
        ),
    )

    _assert_non_stereogenic(molecule, record_index=0)


def test_partial_descendant_group_inversion_keeps_carriers_distinct():
    molecule = _descendant_stereo_molecule("TH", right_value="odd")
    offset = len(molecule.atoms)
    for atomic_number in (15, 9, 17, 35, 53):
        molecule.create_atom(atomic_number=atomic_number)
    for carrier in range(offset + 1, offset + 5):
        molecule.add_bond(offset, carrier, 1.0)

    payload = stereo.get_stereo_payload(molecule)
    group = stereo.StereoGroupRef(
        "gate6-partial",
        stereo.StereoGroupKind.AND,
        3,
    )
    stereo.set_stereo_payload(
        molecule,
        replace(
            payload,
            records=(
                payload.records[0],
                replace(payload.records[1], group=group),
                replace(payload.records[2], group=group),
                replace(
                    _tetrahedral(
                        offset,
                        tuple(
                            _atom(row)
                            for row in range(offset + 1, offset + 5)
                        ),
                    ),
                    group=group,
                ),
            ),
        ),
    )

    assert stereo.validate_stereochemistry(molecule) == (
        stereo.StereoValidationReport(True, ())
    )


def test_current_record_group_does_not_block_its_own_carrier_swap():
    molecule = _molecule(
        (14, 6, 6, 9, 17),
        tuple((0, row, 1.0) for row in range(1, 5)),
    )
    group = stereo.StereoGroupRef(
        "gate6-current",
        stereo.StereoGroupKind.AND,
        1,
    )
    _set_payload(
        molecule,
        (
            replace(
                _tetrahedral(
                    0,
                    tuple(_atom(row) for row in range(1, 5)),
                ),
                group=group,
            ),
        ),
    )

    _assert_non_stereogenic(molecule)


def test_descendant_equivalent_hydrogens_do_not_hide_parent_equivalence():
    molecule = _molecule(
        (14, 6, 6, 17, 35, 1, 9, 1, 9),
        (
            (0, 1, 1.0),
            (0, 2, 1.0),
            (0, 3, 1.0),
            (0, 4, 1.0),
            (1, 5, 1.0),
            (1, 6, 1.0),
            (2, 7, 1.0),
            (2, 8, 1.0),
        ),
        implicit_hydrogens=(0, 1, 1, 0, 0, 0, 0, 0, 0),
    )
    _set_payload(
        molecule,
        (
            _tetrahedral(0, tuple(_atom(row) for row in (1, 2, 3, 4))),
            _tetrahedral(1, (_atom(0), _atom(5), _H, _atom(6))),
            _tetrahedral(
                2,
                (_atom(0), _atom(7), _H, _atom(8)),
                "odd",
            ),
        ),
    )

    report = stereo.validate_stereochemistry(molecule)

    assert {
        issue.record_index
        for issue in report.issues
        if issue.code is stereo.StereoIssueCode.NON_STEREOGENIC
    } == {0, 1, 2}


def _unsupported_semantic_molecule(*, moved):
    atomic_numbers = (14, 6, 6, 9, 17) if moved else (14, 6, 6, 9, 17, 29)
    molecule = _molecule(
        atomic_numbers,
        tuple((0, row, 1.0) for row in range(1, 5)),
    )
    unsupported_anchor = 1 if moved else 5
    _set_payload(
        molecule,
        (
            _tetrahedral(0, tuple(_atom(row) for row in range(1, 5))),
            _unsupported_sp(unsupported_anchor),
        ),
    )
    return molecule


def test_moved_unsupported_semantic_stereo_makes_equivalence_unsupported():
    molecule = _unsupported_semantic_molecule(moved=True)

    report = stereo.validate_stereochemistry(molecule)

    assert not _issues_for(
        report,
        0,
        stereo.StereoIssueCode.NON_STEREOGENIC,
    )
    assert _issues_for(
        report,
        0,
        stereo.StereoIssueCode.UNSUPPORTED,
    )


def test_pointwise_fixed_unrelated_unsupported_stereo_does_not_block_result():
    molecule = _unsupported_semantic_molecule(moved=False)

    report = _assert_non_stereogenic(molecule, record_index=0)

    assert _issues_for(
        report,
        1,
        stereo.StereoIssueCode.UNSUPPORTED,
    )


def test_repeated_public_validation_is_deterministic_and_nonmutating():
    molecule = _hidden_equivalence_molecule(hide_second_oxygen=True)
    payload = stereo.get_stereo_payload(molecule)
    atom_snapshot = tuple(deepcopy(atom.attr_dict) for atom in molecule.atoms)
    active_bond_snapshot = tuple(
        deepcopy(bond.attr_dict) for bond in molecule.bonds
    )
    hidden_bond_snapshot = tuple(
        deepcopy(bond.attr_dict) for bond in molecule._hided_bonds
    )

    first = stereo.validate_stereochemistry(molecule)
    second = stereo.validate_stereochemistry(molecule)

    assert first == second
    assert _issues_for(
        first,
        0,
        stereo.StereoIssueCode.NON_STEREOGENIC,
    )
    assert stereo.get_stereo_payload(molecule) is payload
    assert tuple(atom.attr_dict for atom in molecule.atoms) == atom_snapshot
    assert tuple(bond.attr_dict for bond in molecule.bonds) == active_bond_snapshot
    assert tuple(
        bond.attr_dict for bond in molecule._hided_bonds
    ) == hidden_bond_snapshot


def test_rejected_swap_does_not_enumerate_unrelated_hydrogen_symmetry(
    monkeypatch,
):
    molecule = _descendant_stereo_molecule("TH", right_value="odd")
    molecule.create_atom(atomic_number=6, implicit_hydrogens=4)
    yielded = 0
    original = _NativeSpecTarget.iter_mappings

    def counted(self, query_spec):
        nonlocal yielded
        for mapping in original(self, query_spec):
            yielded += 1
            yield mapping

    monkeypatch.setattr(_NativeSpecTarget, "iter_mappings", counted)

    assert stereo.validate_stereochemistry(molecule) == (
        stereo.StereoValidationReport(True, ())
    )
    assert yielded <= 2


def test_equivalent_methyl_carriers_use_one_hydrogen_slot_mapping(monkeypatch):
    molecule = _molecule(
        (14, 6, 6, 9, 17),
        tuple((0, row, 1.0) for row in range(1, 5)),
        implicit_hydrogens=(0, 3, 3, 0, 0),
    )
    _set_payload(
        molecule,
        (_tetrahedral(0, tuple(_atom(row) for row in range(1, 5))),),
    )
    yielded = 0
    original = _NativeSpecTarget.iter_mappings

    def counted(self, query_spec):
        nonlocal yielded
        for mapping in original(self, query_spec):
            yielded += 1
            yield mapping

    monkeypatch.setattr(_NativeSpecTarget, "iter_mappings", counted)

    _assert_non_stereogenic(molecule)
    assert yielded == 1
