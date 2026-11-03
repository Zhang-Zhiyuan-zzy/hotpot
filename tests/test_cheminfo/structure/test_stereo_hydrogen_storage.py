"""Canonical occurrences retain the existing H consumer/materialization scopes."""
from dataclasses import replace

from hotpot import Molecule
from hotpot.cheminfo import stereo
from hotpot.cheminfo.operations import chemistry
from hotpot.cheminfo.structure._identity import _hydrogen_transport_inputs
from hotpot.cheminfo.structure._fields import FIELD_INDEX


def test_duplicate_H_consumption_is_transient_last_row_and_materialization_updates_every_row():
    mol = Molecule()
    atom = mol.create_atom(atomic_number=6)
    source_h = stereo.HydrogenTransportState(17, 2, False)
    first = stereo.AtomTransportRecord(0, 13, 1, stereo.HydrogenTransportState(71, 2, False),
                                        source_h, "one")
    last = replace(first, isotope=14, radical_electrons=2,
                   current_hydrogens=stereo.HydrogenTransportState(999, 3, True), source_namespace="two")
    payload = stereo.StereoPayload(1, (), (first, last), ())
    stereo.set_stereo_payload(mol, payload)
    assert stereo.get_stereo_payload(mol) == payload
    assert len(mol._atom_table._stereo_facts[0]) == 2
    assert _hydrogen_transport_inputs((atom._uid,), stereo.get_stereo_payload(mol)) == ((atom._uid, 2, 3, True),)
    chemistry._materialize_current_hydrogens(atom)
    materialized = stereo.get_stereo_payload(mol)
    assert tuple(f.current_hydrogens for f in materialized.atom_facts) == (
        stereo.HydrogenTransportState(0, 0, False), stereo.HydrogenTransportState(0, 0, True))
    assert tuple(f.source_hydrogens for f in materialized.atom_facts) == (source_h, source_h)
    assert tuple(f.isotope for f in materialized.atom_facts) == (13, 14)


def test_transport_omitted_H_never_overwrites_legacy_site_total_and_getter_does_not_prepare(monkeypatch):
    mol = Molecule()
    atom = mol.create_atom(atomic_number=6, implicit_hydrogens=4)
    fact = stereo.AtomTransportRecord(0, None, None, stereo.HydrogenTransportState(999, None, None),
                                       None, "declared")
    stereo.set_stereo_payload(mol, stereo.StereoPayload(1, (), (fact,), ()))
    def forbidden(*args, **kwargs):
        raise AssertionError("assembly reached H preparation")
    monkeypatch.setattr(chemistry, "_require_implicit_hydrogens", forbidden)
    assert stereo.get_stereo_payload(mol).atom_facts == (fact,)
    assert mol._atom_table._columns[FIELD_INDEX["implicit_hydrogens"]][0] == 4


def test_dependency_bound_stereo_commit_checks_source_before_its_own_result_invalidation():
    from hotpot.cheminfo.structure._commit import _commit_stereo
    from hotpot.cheminfo.structure._identity import _capture_source
    from hotpot.cheminfo.structure._fields import ATOM_SOURCE_CODES
    from hotpot.cheminfo.structure._results import _import_partial_charges
    mol = Molecule()
    mol.create_atom(atomic_number=6)
    _import_partial_charges(mol, (0.0,), producer=ATOM_SOURCE_CODES["explicit"])
    source = _capture_source(mol, ("bonding_inputs",))
    payload = stereo.StereoPayload(1, (), (stereo.AtomTransportRecord(0, 13, None, None, None, None),), ())
    _commit_stereo(mol, source, payload)
    assert stereo.get_stereo_payload(mol) == payload
    assert mol._atoms[0].isotope == 13
    assert mol._partial_charge_result.invalidated_by
