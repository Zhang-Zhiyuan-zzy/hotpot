from dataclasses import replace

import numpy as np
import pytest

from hotpot.cheminfo import stereo
from hotpot import Molecule
from hotpot.cheminfo.operations import chemistry
from hotpot.cheminfo.structure._stereo import (
    _atom_isotope, _decompose_stereo, _get_stereo_payload,
    _install_stereo_storage, _replace_isotope, _restore_stereo_storage,
    _stereo_storage_state,
)


def _molecule(count=3):
    mol = Molecule()
    for _ in range(count):
        mol.create_atom(atomic_number=6)
    return mol


def _fact(row, isotope=None, *, radical=None, current=None, source=None, namespace=None):
    return stereo.AtomTransportRecord(row, isotope, radical, current, source, namespace)


def _opaque_record():
    return stereo.StereoRecord(
        stereo.StereoAnchor(stereo.StereoAnchorKind.AXIS, (-7, 81, -7)),
        (stereo.StereoCarrier(stereo.StereoCarrierKind.VACANCY, -9),),
        "opaque/future", stereo.StereoGroupRef("", stereo.StereoGroupKind.AND, -(1 << 90)),
        "future-source", "",
    )


def test_bound_occurrences_have_only_canonical_current_owners():
    mol = _molecule()
    first = mol.add_bond(0, 1, 1.0)
    second = mol.add_bond(1, 2, 1.0)
    source_h = stereo.HydrogenTransportState(-3, 1 << 100, None)
    current_h = stereo.HydrogenTransportState(1 << 100, -4, False)
    absent_members = stereo.HydrogenTransportState(None, None, None)
    unbound = _fact(99, -(1 << 100), source=source_h, namespace="unbound")
    missing_edge = stereo.BondTransportRecord((0, 2), "future-direction")
    missing_atom = stereo.BondTransportRecord((-1, 0), "")
    payload = stereo.StereoPayload(
        -(1 << 100), (_opaque_record(),),
        (_fact(2, 1 << 100, radical=-2, current=current_h, source=source_h, namespace=""),
         _fact(0), unbound,
         _fact(2, None, current=absent_members, namespace="other")),
        (stereo.BondTransportRecord((1, 0), "wedge"), missing_edge,
         stereo.BondTransportRecord((0, 1), None), missing_atom,
         stereo.BondTransportRecord((2, 1), "")),
    )

    stereo.set_stereo_payload(mol, payload)

    assert "_stereo_payload" not in mol.__dict__
    assert mol._stereo_schema == -(1 << 100)
    assert mol._stereo_records is payload.records
    assert mol._stereo_source == (
        ((mol.atoms[2]._uid, 0, source_h), (mol.atoms[0]._uid, 0, None), unbound,
         (mol.atoms[2]._uid, 1, None)),
        ((first._uid, 0), missing_edge, (first._uid, 1), missing_atom, (second._uid, 0)),
    )
    assert mol._stereo_source[0][2] is unbound
    assert mol._stereo_source[1][1] is missing_edge
    assert mol._atom_table._stereo_facts[:3] == [
        ((None, None, False, None, None, None, None),),
        (),
        ((1 << 100, -2, True, 1 << 100, -4, False, ""),
         (None, None, True, None, None, None, "other")),
    ]
    assert first._display_facts == (
        (mol.atoms[1]._uid, mol.atoms[0]._uid, "wedge"),
        (mol.atoms[0]._uid, mol.atoms[1]._uid, None),
    )
    assert second._display_facts == ((mol.atoms[2]._uid, mol.atoms[1]._uid, ""),)
    assembled = stereo.get_stereo_payload(mol)
    assert assembled == payload
    assert type(assembled.schema_version) is int
    assert type(assembled.atom_facts[0].isotope) is int
    assert assembled.atom_facts[0].current_hydrogens.no_implicit is False
    assert assembled.atom_facts[0].source_hydrogens is source_h
    assert assembled.atom_facts[2] is unbound
    assert assembled.bond_facts[1] is missing_edge



def test_h_container_presence_and_empty_source_namespaces_round_trip_independently():
    mol = _molecule(1)
    empty = stereo.HydrogenTransportState(None, None, None)
    zero = stereo.HydrogenTransportState(0, 0, False)
    blocked = stereo.HydrogenTransportState(0, 0, True)
    payload = stereo.StereoPayload(1, (), tuple(
        _fact(0, current=current, source=source, namespace=namespace)
        for current, source, namespace in (
            (None, empty, None), (empty, None, ""),
            (zero, blocked, "future"), (blocked, zero, ""),
        )
    ), ())

    stereo.set_stereo_payload(mol, payload)

    assembled = stereo.get_stereo_payload(mol)
    assert assembled == payload
    assert assembled.atom_facts[0].current_hydrogens is None
    assert assembled.atom_facts[0].source_hydrogens is empty
    assert assembled.atom_facts[1].current_hydrogens == empty
    assert assembled.atom_facts[1].source_hydrogens is None
    assert assembled.atom_facts[2].current_hydrogens.no_implicit is False
    assert assembled.atom_facts[3].current_hydrogens.no_implicit is True
    assert tuple(fact.source_namespace for fact in assembled.atom_facts) == (None, "", "future", "")


def test_assembly_reads_current_atom_and_bond_occurrence_owners():
    mol = _molecule(2)
    bond = mol.add_bond(0, 1, 1.0)
    payload = stereo.StereoPayload(
        47, (), (_fact(1, 13, current=stereo.HydrogenTransportState(None, 0, None)),),
        (stereo.BondTransportRecord((1, 0), "NONE"),),
    )
    stereo.set_stereo_payload(mol, payload)
    mol._atom_table._stereo_facts[1] = ((17, 1, True, 9, 2, True, "changed"),)
    bond._display_facts = ((mol.atoms[0]._uid, mol.atoms[1]._uid, "future"),)

    assert stereo.get_stereo_payload(mol) == stereo.StereoPayload(
        47, (), (_fact(1, 17, radical=1,
                      current=stereo.HydrogenTransportState(9, 2, True), namespace="changed"),),
        (stereo.BondTransportRecord((0, 1), "future"),),
    )
    assert payload.atom_facts[0].isotope == 13
    assert payload.bond_facts[0].atom_indices == (1, 0)
    assert "_stereo_payload" not in mol.__dict__


def test_decompose_is_pure_and_fresh_install_preserves_opaque_unbound_values():
    mol = _molecule(0)
    payload = stereo.StereoPayload(
        47, (_opaque_record(),), (_fact(81, 13),),
        (stereo.BondTransportRecord((81, -9), "unknown"),),
    )
    before = _stereo_storage_state(mol)

    schema, records, source, atom_rows, bond_rows = _decompose_stereo(mol, payload)

    assert _stereo_storage_state(mol) == before
    assert (schema, records, source, atom_rows, bond_rows) == (
        47, payload.records, (payload.atom_facts, payload.bond_facts), (), (),
    )
    _install_stereo_storage(mol, payload)
    assert _get_stereo_payload(mol) == payload
    assert mol._stereo_source[0][0] is payload.atom_facts[0]
    assert mol._stereo_source[1][0] is payload.bond_facts[0]


def test_get_and_set_do_not_prepare_chemistry(monkeypatch):
    mol = _molecule(1)
    payload = stereo.StereoPayload(
        1, (), (_fact(0, current=stereo.HydrogenTransportState(999, None, None)),), (),
    )

    def forbidden(*args, **kwargs):
        pytest.fail("raw stereo storage prepared chemistry")

    for name in ("_calculate_implicit_hydrogens", "_require_implicit_hydrogens",
                 "_require_aromatic_representation"):
        monkeypatch.setattr(chemistry, name, forbidden)
    stereo.set_stereo_payload(mol, payload)
    assert stereo.get_stereo_payload(mol) == payload
    stereo.set_stereo_payload(mol, stereo.get_stereo_payload(mol))
    assert stereo.get_stereo_payload(mol) == payload


def test_display_occurrences_are_separate_from_dative_direction():
    mol = _molecule(2)
    bond = mol.add_bond(0, 1, 1.0, bond_kind="dative", bond_direction="atom2_to_atom1")
    payload = stereo.StereoPayload(1, (), (), (
        stereo.BondTransportRecord((1, 0), "ENDDOWNRIGHT"),
        stereo.BondTransportRecord((0, 1), ""),
        stereo.BondTransportRecord((1, 0), None),
    ))

    stereo.set_stereo_payload(mol, payload)

    assert bond.bond_direction == "atom2_to_atom1"
    assert len(bond._display_facts) == 3
    assert stereo.get_stereo_payload(mol).bond_facts == payload.bond_facts


def test_none_clears_all_canonical_owners_and_isotope_assignment_starts_schema1():
    mol = _molecule(2)
    bond = mol.add_bond(0, 1, 1.0)
    stereo.set_stereo_payload(mol, stereo.StereoPayload(
        47, (_opaque_record(),), (_fact(0, 13),),
        (stereo.BondTransportRecord((1, 0), "future"),),
    ))

    stereo.set_stereo_payload(mol, None)

    assert stereo.get_stereo_payload(mol) is None
    assert mol._stereo_schema is None
    assert mol._stereo_records == ()
    assert mol._stereo_source == ((), ())
    assert mol._atom_table._stereo_facts[:2] == [(), ()]
    assert bond._display_facts == ()
    assert not hasattr(mol.atoms[0], "isotope")
    mol.atoms[0].isotope = 0
    assert stereo.get_stereo_payload(mol) == stereo.StereoPayload(1, (), (_fact(0, 0),), ())


def test_storage_snapshot_restores_exact_layout_and_every_bond_display_reference():
    mol = _molecule()
    first = mol.add_bond(0, 1, 1.0)
    second = mol.add_bond(1, 2, 1.0)
    payload = stereo.StereoPayload(47, (_opaque_record(),), (_fact(0, 13),),
                                  (stereo.BondTransportRecord((1, 0), ""),))
    stereo.set_stereo_payload(mol, payload)
    state = _stereo_storage_state(mol)
    private_plane = mol._atom_table._stereo_facts
    private_row = private_plane[0]
    assert state[3] == ((first, first._display_facts), (second, second._display_facts))
    mol._stereo_schema, mol._stereo_records, mol._stereo_source = 1, (), ((), ())
    first._display_facts = ()
    second._display_facts = ((mol.atoms[2]._uid, mol.atoms[1]._uid, "changed"),)

    _restore_stereo_storage(mol, state)

    assert mol._stereo_records is state[1]
    assert mol._stereo_source is state[2]
    assert first._display_facts is state[3][0][1]
    assert second._display_facts is state[3][1][1]
    assert mol._atom_table._stereo_facts is private_plane
    assert mol._atom_table._stereo_facts[0] is private_row
    assert stereo.get_stereo_payload(mol) == payload


@pytest.mark.parametrize("isotopes", [(None,), (None, None), (13,), (13, 13), (-(1 << 100),) * 2])
def test_isotope_scalar_projects_equal_occurrences(isotopes):
    mol = _molecule(1)
    stereo.set_stereo_payload(mol, stereo.StereoPayload(
        1, (), tuple(_fact(0, value) for value in isotopes), (),
    ))

    assert _atom_isotope(mol.atoms[0]) == isotopes[0]
    assert mol.atoms[0].isotope == isotopes[0]
    assert "isotope" not in mol.atoms[0].__dict__


@pytest.mark.parametrize("isotopes", [(13, 15), (None, 13), (0, None)])
def test_conflicting_duplicate_isotopes_fail_only_at_scalar_consumption(isotopes):
    mol = _molecule(1)
    payload = stereo.StereoPayload(1, (), tuple(_fact(0, value) for value in isotopes), ())

    stereo.set_stereo_payload(mol, payload)

    assert stereo.get_stereo_payload(mol) == payload
    with pytest.raises(ValueError, match="Conflicting isotope declarations for atom row 0"):
        _atom_isotope(mol.atoms[0])
    with pytest.raises(ValueError, match="Conflicting isotope declarations"):
        _ = mol.atoms[0].isotope
    assert stereo.get_stereo_payload(mol) == payload


def test_missing_isotope_is_distinct_from_present_none_and_detached_read_uses_snapshot():
    mol = _molecule(1)
    atom = mol.atoms[0]
    with pytest.raises(AttributeError, match="isotope"):
        _atom_isotope(atom)
    stereo.set_stereo_payload(mol, stereo.StereoPayload(1, (), (_fact(0, None),), ()))
    assert atom.isotope is None
    atom.isotope = 1 << 100

    mol.remove_atom(atom)

    assert atom.isotope == 1 << 100
    assert atom._detached_table._stereo_facts[0][0][0] == 1 << 100
    assert "isotope" not in atom.__dict__


def test_direct_isotope_assignment_updates_every_occurrence_and_retains_other_fields():
    mol = _molecule(2)
    source = stereo.HydrogenTransportState(7, 2, None)
    current = stereo.HydrogenTransportState(999, None, False)
    payload = stereo.StereoPayload(47, (_opaque_record(),), (
        _fact(1, 13, radical=2, current=current, source=source, namespace=""),
        _fact(0, 12), _fact(1, 15, source=source, namespace="other"),
    ), ())
    stereo.set_stereo_payload(mol, payload)

    mol.atoms[1].isotope = np.int64(17)

    expected = replace(payload, atom_facts=tuple(
        replace(fact, isotope=17) if fact.atom == 1 else fact for fact in payload.atom_facts))
    assert stereo.get_stereo_payload(mol) == expected
    assert mol.atoms[1].isotope == 17
    assert type(mol.atoms[1].isotope) is int
    assert all(fact.source_hydrogens is source for fact in
               stereo.get_stereo_payload(mol).atom_facts if fact.atom == 1)
    assert "isotope" not in mol.atoms[1].__dict__


def test_replace_isotope_is_pure_preserves_order_and_appends_only_when_absent():
    history = stereo.HydrogenTransportState(7, 3, True)
    payload = stereo.StereoPayload(47, (_opaque_record(),), (
        _fact(2, 13, radical=-1, source=history, namespace="source"),
        _fact(0), _fact(2, None),
    ), (stereo.BondTransportRecord((81, 0), "opaque"),))

    changed = _replace_isotope(payload, 2, None)
    appended = _replace_isotope(payload, 1, -(1 << 100))

    assert changed == replace(payload, atom_facts=(
        replace(payload.atom_facts[0], isotope=None), payload.atom_facts[1], payload.atom_facts[2]))
    assert changed.atom_facts[0].source_hydrogens is history
    assert payload.atom_facts[0].isotope == 13
    assert appended.atom_facts[:-1] == payload.atom_facts
    assert appended.atom_facts[-1] == _fact(1, -(1 << 100))
    assert appended.schema_version == 47
    assert appended.records is payload.records
    assert appended.bond_facts is payload.bond_facts
    assert _replace_isotope(None, 2, 0) == stereo.StereoPayload(1, (), (_fact(2, 0),), ())


@pytest.mark.parametrize("change", [
    {"isotope": True}, {"radical_electrons": False},
    {"current_hydrogens": stereo.HydrogenTransportState(True, None, None)},
    {"current_hydrogens": stereo.HydrogenTransportState(None, 0, 0)},
])
def test_boolean_integer_confusion_is_rejected_without_changing_canonical_owners(change):
    mol = _molecule(1)
    payload = stereo.StereoPayload(1, (), (_fact(0, 13),), ())
    stereo.set_stereo_payload(mol, payload)
    row = mol._atom_table._stereo_facts[0]
    source = mol._stereo_source

    with pytest.raises(TypeError):
        stereo.set_stereo_payload(mol, replace(payload, atom_facts=(replace(payload.atom_facts[0], **change),)))

    assert mol._atom_table._stereo_facts[0] is row
    assert mol._stereo_source is source
    assert stereo.get_stereo_payload(mol) == payload


def test_broken_bound_uid_is_an_integrity_error_instead_of_an_unbound_fallback():
    mol = _molecule(1)
    stereo.set_stereo_payload(mol, stereo.StereoPayload(1, (), (_fact(0, 13),), ()))
    mol._stereo_source = (((-1, 0, None),), ())

    with pytest.raises(KeyError):
        stereo.get_stereo_payload(mol)
