"""Independent chemical inventories, source isolation and atomic action failures."""

from dataclasses import replace

import pytest
from rdkit import Chem

import hotpot as hp
from hotpot.cheminfo.operations.assembly import AtomLink, RingWedge
from hotpot.cheminfo.operations.assembly import action_func as actions
from hotpot.cheminfo.operations.conversion import from_rdmol
from hotpot.cheminfo.operations.chemistry import _set_stereo_payload
from hotpot.cheminfo.structure._dependencies import _dependency_state
from hotpot.cheminfo.structure._fields import ATOM_SOURCE_CODES, FIELD_INDEX
from hotpot.cheminfo.structure._stereo import _get_stereo_payload
from hotpot.cheminfo.structure.bond_kind import BondKind
from hotpot.cheminfo.structure.stereo import (
    AtomTransportRecord, HydrogenTransportState, StereoAnchor, StereoAnchorKind,
    StereoCarrier, StereoCarrierKind, StereoPayload, StereoRecord,
)
from tests.test_cheminfo._stage5_fixtures import factual_state, immutable_snapshot


def _molecule(numbers, edges=(), counts=None):
    mol = hp.Molecule()
    for i, number in enumerate(numbers):
        attrs = {'atomic_number': number, 'formal_charge': 0}
        if counts is not None:
            attrs['implicit_hydrogens'] = counts[i]
        mol.create_atom(**attrs)
    for first, second, order in edges:
        mol.add_bond(first, second, order)
    return mol


def _rdkit(smiles, graph=False, keep_h=False):
    params = Chem.SmilesParserParams()
    params.removeHs = not keep_h
    source = Chem.MolFromSmiles(smiles, params)
    if graph:
        source = Chem.AddHs(source)
    return from_rdmol(source, hp.Molecule())


def _graph(number, count, user=False):
    return _molecule([number] + [1] * count, [(0, i, 1) for i in range(1, count + 1)],
                     ([4 if number == 6 else 2] + [0] * count) if user else None)


def _state(mol):
    return (factual_state(mol), immutable_snapshot(_dependency_state(mol._atom_table)),
            tuple(id(a) for a in mol.atoms), tuple(id(b) for b in mol.bonds))


def _heavy(mol):
    return tuple(atom for atom in mol.atoms if atom.atomic_number != 1)


def _count(atom):
    return int(atom.mol._atom_table._columns[FIELD_INDEX['implicit_hydrogens']][atom.idx])


def _source(atom):
    return int(atom.mol._atom_table._sources[atom.idx, FIELD_INDEX['implicit_hydrogens']])


def _site_facts(atom):
    payload = _get_stereo_payload(atom.mol)
    return () if payload is None else tuple(f for f in payload.atom_facts if f.atom == atom.idx)


def _set_current(atom, implicit, bracket, no_implicit, *, source=None, isotope=0):
    payload = _get_stereo_payload(atom.mol) or StereoPayload(1, (), (), ())
    current = HydrogenTransportState(implicit, bracket, no_implicit)
    fact = AtomTransportRecord(atom.idx, isotope, 0, current, source or current, 'fixture')
    _set_stereo_payload(atom.mol, replace(payload,
        atom_facts=tuple(f for f in payload.atom_facts if f.atom != atom.idx) + (fact,)))


def _set_isotope(atom, isotope):
    payload = _get_stereo_payload(atom.mol) or StereoPayload(1, (), (), ())
    facts = tuple(replace(f, isotope=isotope) if f.atom == atom.idx else f for f in payload.atom_facts)
    if not any(f.atom == atom.idx for f in facts):
        facts += (AtomTransportRecord(atom.idx, isotope, None, None, None, 'fixture'),)
    _set_stereo_payload(atom.mol, replace(payload, atom_facts=facts))


def _isotope(atom):
    return next((f.isotope for f in _site_facts(atom)), 0)


def _assert_product_sites(atoms, counts):
    assert tuple(_count(atom) for atom in atoms) == counts
    assert tuple(_source(atom) for atom in atoms) == (ATOM_SOURCE_CODES['explicit'],) * len(atoms)
    assert tuple(atom.formal_charge for atom in atoms) == (0,) * len(atoms)
    for atom, count in zip(atoms, counts):
        graph = len(atom.hydrogens)
        assert int(atom.mol._atom_table._presence[atom.idx]) & (1 << FIELD_INDEX['implicit_hydrogens'])
        assert graph <= count
        for fact in _site_facts(atom):
            state = fact.current_hydrogens
            if state is not None:
                assert state.implicit_hydrogens + state.bracket_explicit_hydrogens == count - graph
                assert not state.no_implicit or state.implicit_hydrogens == 0


_LINK_VARIANTS = (
    'inferred', 'user', 'openbabel', 'rdkit', 'graph_full_inferred',
    'graph_partial_inferred', 'graph_full_rdkit', 'graph_partial_rdkit', 'graph_full_user',
)


def _link_inputs(variant):
    if variant == 'inferred':
        return _molecule([6]), _molecule([8])
    if variant == 'user':
        return _molecule([6], counts=[4]), _molecule([8], counts=[2])
    if variant == 'openbabel':
        return hp.read_mol('C'), hp.read_mol('O')
    if variant == 'rdkit':
        return _rdkit('C'), _rdkit('O')
    if variant == 'graph_full_inferred':
        return _graph(6, 4), _graph(8, 2)
    if variant == 'graph_partial_inferred':
        return _graph(6, 1), _graph(8, 1)
    if variant == 'graph_full_rdkit':
        return _rdkit('C', graph=True), _rdkit('O', graph=True)
    if variant == 'graph_partial_rdkit':
        return _rdkit('[H]C', keep_h=True), _rdkit('[H]O', keep_h=True)
    return _graph(6, 4, user=True), _graph(8, 2, user=True)


@pytest.mark.parametrize('variant', _LINK_VARIANTS)
def test_link_methanol_nine_current_representations_and_originals(variant):
    original, fragment = _link_inputs(variant)
    before, fragment_before = _state(original), _state(fragment)
    product = original.copy()
    carbon = next(a for a in product.atoms if a.atomic_number == 6)
    oxygen = next(a for a in fragment.atoms if a.atomic_number == 8)
    source_carbon = tuple(f.source_hydrogens for f in _site_facts(carbon))
    source_oxygen = tuple(f.source_hydrogens for f in _site_facts(oxygen))
    assert actions.atom_link_atom_action(product, [carbon.idx], fragment, [oxygen.idx]) is product
    carbon, oxygen = _heavy(product)
    _assert_product_sites((carbon, oxygen), (3, 1))
    assert len(carbon.heavy_neighbours) == len(oxygen.heavy_neighbours) == 1
    assert product.bond(carbon.idx, oxygen.idx).bond_kind is BondKind.SINGLE
    expected_graph = (3, 1) if 'full' in variant else (1, 1) if 'partial' in variant else (0, 0)
    assert (len(carbon.hydrogens), len(oxygen.hydrogens)) == expected_graph
    assert tuple(f.source_hydrogens for f in _site_facts(carbon)) == source_carbon
    assert tuple(f.source_hydrogens for f in _site_facts(oxygen)) == source_oxygen
    assert not any(a.atomic_number == 1 and not a.neighbours for a in product.atoms)
    assert _state(original) == before
    assert _state(fragment) == fragment_before


def test_actual_graft_preserves_unprepared_frame_and_fragment_fact_states():
    frame, fragment = _molecule([6]), _molecule([8])
    frame_before, fragment_before = _state(frame), _state(fragment)
    products = AtomLink(fragment, (0,)).graft(frame)
    assert len(products) == 1
    _assert_product_sites(_heavy(next(iter(products.values()))), (3, 1))
    assert _state(frame) == frame_before
    assert _state(fragment) == fragment_before


@pytest.mark.parametrize('graph', [False, True])
def test_sequential_ethane_to_ethene_to_ethyne_consumes_caps(graph):
    mol = _rdkit('CC', graph=True) if graph else hp.read_mol('CC')
    atoms = _heavy(mol)
    for order, hydrogen in ((2, 2), (3, 1)):
        actions.bond_order_add(mol, [atoms[0].idx, atoms[1].idx], hp.Molecule(), [])
        _assert_product_sites(atoms, (hydrogen, hydrogen))
        assert mol.bond(atoms[0].idx, atoms[1].idx).bond_order == order
        assert tuple(len(atom.hydrogens) for atom in atoms) == ((hydrogen, hydrogen) if graph else (0, 0))
    before = _state(mol)
    with pytest.raises(ValueError, match='ASSEMBLY.HYDROGEN_INVENTORY'):
        actions.bond_order_add(mol, [atoms[0].idx, atoms[1].idx], hp.Molecule(), [])
    assert _state(mol) == before


@pytest.mark.parametrize('smiles,target,expected', [('C', 7, (3,)), ('CC', 8, (3, 1))])
@pytest.mark.parametrize('graph', [False, True])
def test_element_replacement_completes_only_selected_site(smiles, target, expected, graph):
    mol = _rdkit(smiles, graph=True) if graph else hp.read_mol(smiles)
    atoms = _heavy(mol)
    old_source = _source(atoms[0])
    actions.atom_replace(mol, [atoms[-1].idx], _molecule([target]), [])
    assert tuple(_count(atom) for atom in atoms) == (expected if not graph else ((3,) if target == 7 else (0, 1)))
    _assert_product_sites((atoms[-1],), (expected[-1],))
    assert atoms[-1].atomic_number == target
    if len(atoms) > 1:
        assert _source(atoms[0]) == old_source
    if graph:
        assert len(atoms[-1].hydrogens) == expected[-1]
        assert not any(a.atomic_number == 1 and not a.neighbours for a in mol.atoms)


@pytest.mark.parametrize('graph', [False, True])
def test_shoulder_butene_deletes_parent_caps_and_completes_fragment_caps(graph):
    mol = _rdkit('CCCC', graph=True) if graph else hp.read_mol('CCCC')
    fragment = _rdkit('C=C', graph=True) if graph else hp.read_mol('C=C')
    old = _heavy(mol)
    removed_ids = {id(a) for a in (old[1], old[2], *old[1].hydrogens, *old[2].hydrogens)}
    fragment_before = _state(fragment)
    actions.shoulder_bond_action(mol, [old[1].idx, old[2].idx], fragment, [0, 1])
    assert all(id(a) not in removed_ids for a in mol.atoms)
    left, right, first, second = _heavy(mol)
    _assert_product_sites((first, second), (1, 1))
    assert mol.bond(first.idx, second.idx).bond_kind is BondKind.DOUBLE
    assert {id(a) for a in first.heavy_neighbours} == {id(left), id(second)}
    assert {id(a) for a in second.heavy_neighbours} == {id(right), id(first)}
    if graph:
        assert sum(a.atomic_number == 1 for a in mol.atoms) == 8
        assert (len(left.hydrogens), len(right.hydrogens), len(first.hydrogens), len(second.hydrogens)) == (3, 3, 1, 1)
    else:
        assert tuple(_count(a) for a in (left, first, second, right)) == (3, 1, 1, 3)
    assert not any(a.atomic_number == 1 and not a.neighbours for a in mol.atoms)
    assert _state(fragment) == fragment_before


@pytest.mark.parametrize('graph', [False, True])
def test_ring_retains_original_terminal_identity_isotope_charge_and_source_ledger(graph):
    mol, fragment = _rdkit('C[13CH3]', graph=graph), _rdkit('C1CC1', graph=graph)
    retained = mol.atoms[1]
    source = tuple(f.source_hydrogens for f in _site_facts(retained))
    before = _state(fragment)
    actions.ring_wedge(mol, [retained.idx], fragment, [0])
    assert any(a is retained for a in mol.atoms)
    assert retained.atomic_number == 6 and retained.formal_charge == 0
    assert _isotope(retained) == 13
    assert tuple(f.source_hydrogens for f in _site_facts(retained)) == source
    _assert_product_sites((retained,), (1,))
    assert len(retained.heavy_neighbours) == 3
    assert len(_heavy(mol)) == 4
    assert len(mol.bonds) == (12 if graph else 4)
    if graph:
        assert sum(a.atomic_number == 1 for a in mol.atoms) == 8
    assert not any(a.atomic_number == 1 and not a.neighbours for a in mol.atoms)
    assert _state(fragment) == before


def test_actual_ring_graft_selects_fully_graph_h_sites_and_preserves_input():
    frame, fragment = _rdkit('CC', graph=True), _rdkit('C1CC1', graph=True)
    before, fragment_before = _state(frame), _state(fragment)
    products = RingWedge(fragment, (0,)).graft(frame)
    assert products
    for product in products.values():
        assert len(_heavy(product)) == 4
        assert sum(a.atomic_number == 1 for a in product.atoms) == 8
        assert sorted(len(a.heavy_neighbours) for a in _heavy(product)) == [1, 2, 2, 3]
    assert _state(frame) == before and _state(fragment) == fragment_before


def test_bracket_consumption_preserves_noimplicit_and_immutable_source():
    frame, fragment = _rdkit('[CH4]'), _rdkit('[OH2]')
    sources = (_site_facts(frame.atoms[0])[0].source_hydrogens,
               _site_facts(fragment.atoms[0])[0].source_hydrogens)
    actions.atom_link_atom_action(frame, [0], fragment, [0])
    for atom, count, source in zip(_heavy(frame), (3, 1), sources):
        fact = _site_facts(atom)[0]
        assert fact.current_hydrogens == HydrogenTransportState(0, count, True)
        assert fact.source_hydrogens == source


def test_noimplicit_element_completion_retains_only_declared_h_and_all_occurrences():
    mol = _molecule([6], counts=[0])
    atom = mol.atoms[0]
    _set_current(atom, 0, 0, True)
    payload = _get_stereo_payload(mol)
    _set_stereo_payload(mol, replace(payload, atom_facts=payload.atom_facts * 2))
    sources = tuple(f.source_hydrogens for f in _site_facts(atom))
    actions.atom_replace(mol, [0], _molecule([7]), [])
    assert _count(atom) == 0
    assert tuple(f.current_hydrogens for f in _site_facts(atom)) == (HydrogenTransportState(0, 0, True),) * 2
    assert tuple(f.source_hydrogens for f in _site_facts(atom)) == sources


def test_bracket_capacity_reduces_without_refilling_from_history():
    mol = _rdkit('[CH4]')
    atom = mol.atoms[0]
    source = _site_facts(atom)[0].source_hydrogens
    actions.atom_replace(mol, [0], _molecule([7]), [])
    assert _count(atom) == 3
    assert _site_facts(atom)[0].current_hydrogens == HydrogenTransportState(0, 3, True)
    assert _site_facts(atom)[0].source_hydrogens == source


def test_ordinary_graph_h_removed_before_deuterium():
    mol = _graph(6, 4)
    carbon = mol.atoms[0]
    for hydrogen in carbon.hydrogens[1:]:
        _set_isotope(hydrogen, 2)
    actions.atom_replace(mol, [0], _molecule([7]), [])
    assert len(carbon.hydrogens) == 3
    assert tuple(_isotope(h) for h in carbon.hydrogens) == (2, 2, 2)


def test_distinct_isotope_cap_products_fail_atomically():
    mol = _graph(6, 4)
    for hydrogen, isotope in zip(mol.atoms[0].hydrogens, (2, 2, 3, 3)):
        _set_isotope(hydrogen, isotope)
    before = _state(mol)
    with pytest.raises(ValueError, match='ASSEMBLY.ISOTOPE_AMBIGUITY'):
        actions.atom_replace(mol, [0], _molecule([7]), [])
    assert _state(mol) == before


def _stereo(mol, anchor, carrier=None, descriptor='uninterpreted-fixture'):
    payload = _get_stereo_payload(mol) or StereoPayload(1, (), (), ())
    record = StereoRecord(StereoAnchor(StereoAnchorKind.ATOM, (anchor,)),
        () if carrier is None else (StereoCarrier(StereoCarrierKind.ATOM, carrier),),
        descriptor, None, 'fixture', descriptor)
    _set_stereo_payload(mol, replace(payload, records=payload.records + (record,)))
    return record


@pytest.mark.parametrize('anchor,carrier', [(0, None), (1, 0)])
def test_affected_anchor_or_carrier_stereo_rejected_without_partial_preparation(anchor, carrier):
    mol = _molecule([6, 6], [(0, 1, 1)])
    _stereo(mol, anchor, carrier)
    before = _state(mol)
    with pytest.raises(ValueError, match='ASSEMBLY.STEREO_TRANSFORMATION_UNDEFINED'):
        actions.atom_link_atom_action(mol, [0], _molecule([8]), [0])
    assert _state(mol) == before


def test_unaffected_stereo_and_unrelated_authoritative_h_are_preserved():
    mol = _molecule([6, 6, 8], [(0, 1, 1)], counts=[3, 3, 7])
    remote = mol.atoms[2]
    record = _stereo(mol, 2)
    before = remote._uid, _count(remote), _source(remote), _site_facts(remote)
    actions.atom_link_atom_action(mol, [0], _molecule([8]), [0])
    assert (remote._uid, _count(remote), _source(remote), _site_facts(remote)) == before
    assert record in _get_stereo_payload(mol).records


@pytest.mark.parametrize('count', [-1, 1.5, float('nan')])
def test_invalid_supplied_h_is_a_named_atomic_inventory_failure(count):
    mol = _molecule([6], counts=[count])
    before = _state(mol)
    with pytest.raises(ValueError, match='ASSEMBLY.HYDROGEN_INVENTORY'):
        actions.atom_link_atom_action(mol, [0], _molecule([8]), [0])
    assert _state(mol) == before


def test_contradictory_current_and_observable_materialized_import_fail_atomically():
    mol = _rdkit('C')
    _set_current(mol.atoms[0], 3, 0, False)
    before = _state(mol)
    with pytest.raises(ValueError, match='ASSEMBLY.HYDROGEN_INVENTORY'):
        actions.atom_link_atom_action(mol, [0], _molecule([8]), [0])
    assert _state(mol) == before
    mol = _rdkit('C', graph=True)
    mol.atoms[0].implicit_hydrogens = 8
    before = _state(mol)
    with pytest.raises(ValueError, match='ASSEMBLY.HYDROGEN_INVENTORY'):
        actions.atom_link_atom_action(mol, [0], _molecule([8]), [0])
    assert _state(mol) == before


@pytest.mark.parametrize('action', ['link', 'bond', 'replace', 'shoulder', 'ring'])
@pytest.mark.parametrize('failure', [RuntimeError, KeyboardInterrupt])
def test_whole_action_rolls_back_after_first_product_h_commit(monkeypatch, action, failure):
    if action == 'link':
        mol, frag, fn, hit, points = _molecule([6]), _molecule([8]), actions.atom_link_atom_action, [0], [0]
    elif action == 'bond':
        mol, frag, fn, hit, points = _molecule([6, 6], [(0, 1, 1)]), hp.Molecule(), actions.bond_order_add, [0, 1], []
    elif action == 'replace':
        mol, frag, fn, hit, points = _graph(6, 4), _molecule([7]), actions.atom_replace, [0], []
    elif action == 'shoulder':
        mol, frag, fn, hit, points = _rdkit('CCCC', graph=True), _rdkit('C=C', graph=True), actions.shoulder_bond_action, [1, 2], [0, 1]
    else:
        mol, frag, fn, hit, points = _rdkit('C[13CH3]', graph=True), _rdkit('C1CC1', graph=True), actions.ring_wedge, [1], [0]
    before, frag_before = _state(mol), _state(frag)
    commit = actions._write_hydrogens
    def fail_after_commit(*args):
        commit(*args)
        raise failure('injected after factual product H commit')
    monkeypatch.setattr(actions, '_write_hydrogens', fail_after_commit)
    with pytest.raises(failure, match='injected after factual product H commit'):
        fn(mol, hit, frag, points)
    assert _state(mol) == before
    assert _state(frag) == frag_before


def test_link_rolls_back_merge_when_bond_creation_fails(monkeypatch):
    mol, frag = _molecule([6]), _molecule([8])
    before = _state(mol)
    def fail(*args, **kwargs):
        raise RuntimeError('injected new bond failure')
    monkeypatch.setattr(hp.Molecule, 'add_bond', fail)
    with pytest.raises(RuntimeError, match='injected new bond failure'):
        actions.atom_link_atom_action(mol, [0], frag, [0])
    assert _state(mol) == before


def test_declared_aromatic_nh_cap_is_consumed_without_changing_source_history():
    mol, fragment = _rdkit('c1cc[nH]c1'), _molecule([8])
    nitrogen = next(a for a in mol.atoms if a.atomic_number == 7)
    original_transport = _site_facts(nitrogen)[0]
    source = original_transport.source_hydrogens
    actions.atom_link_atom_action(mol, [nitrogen.idx], fragment, [0])
    oxygen = next(a for a in mol.atoms if a.atomic_number == 8)
    _assert_product_sites((nitrogen, oxygen), (0, 1))
    assert _site_facts(nitrogen)[0].current_hydrogens == replace(
        original_transport.current_hydrogens, implicit_hydrogens=0, bracket_explicit_hydrogens=0)
    assert _site_facts(nitrogen)[0].source_hydrogens == source


def test_shoulder_preserves_outside_dative_orientation_metadata_and_constraint():
    mol = _molecule([6, 6, 6], [(1, 2, 1)])
    outside = mol.add_bond(0, 1, 1., bond_kind=BondKind.DATIVE,
        bond_direction='atom2_to_atom1', bond_source='fixture',
        bond_source_metadata={'meaning': 'preserve donor orientation'})
    outside.constraint = 2.5
    outside.id = 23
    retained = mol.atoms[0]
    actions.shoulder_bond_action(mol, [1, 2], _molecule([6, 6], [(0, 1, 2)]), [0, 1])
    transferred = next(b for b in retained.bonds if b.bond_kind is BondKind.DATIVE)
    assert transferred.atom1 is retained
    assert transferred.bond_direction == 'atom2_to_atom1'
    assert transferred.bond_order == 1.
    assert transferred.bond_source == 'fixture'
    assert transferred.bond_source_metadata == {'meaning': 'preserve donor orientation'}
    assert transferred.constraint is True
    assert tuple(transferred.attrs) == (1., 2.5, 23.)
    _assert_product_sites(tuple(a for a in _heavy(mol) if a is not retained), (2, 2))


def test_ring_adopts_aromatic_scaffold_at_the_retained_parent_site():
    mol, fragment = _rdkit('C[13CH3]'), _rdkit('c1ccccc1')
    retained = mol.atoms[1]
    source = _site_facts(retained)[0].source_hydrogens
    actions.ring_wedge(mol, [retained.idx], fragment, [0])
    assert any(a is retained for a in mol.atoms)
    assert bool(mol._atom_table._columns[FIELD_INDEX['is_aromatic']][retained.idx])
    assert int(mol._atom_table._sources[retained.idx, FIELD_INDEX['is_aromatic']]) == ATOM_SOURCE_CODES['explicit']
    mol.calc_implicit_hydrogens()
    assert retained.is_aromatic
    assert _isotope(retained) == 13
    assert _site_facts(retained)[0].source_hydrogens == source
    _assert_product_sites((retained,), (0,))
    assert len(_heavy(mol)) == 7
    assert sorted(len(a.heavy_neighbours) for a in _heavy(mol)) == [1, 2, 2, 2, 2, 2, 3]
    assert tuple(sorted(_count(a) for a in _heavy(mol))) == (0, 1, 1, 1, 1, 1, 3)


def test_shoulder_adopts_parent_aromatic_flags_before_completing_both_anchors():
    mol, fragment = _rdkit('c1ccccc1'), _rdkit('C=C')
    fragment_before = _state(fragment)
    source = tuple(_site_facts(atom)[0].source_hydrogens for atom in fragment.atoms)
    actions.shoulder_bond_action(mol, [0, 1], fragment, [0, 1])
    replacements = mol.atoms[-2:]
    assert tuple(bool(mol._atom_table._columns[FIELD_INDEX['is_aromatic']][atom.idx])
                 for atom in replacements) == (True, True)
    assert tuple(int(mol._atom_table._sources[atom.idx, FIELD_INDEX['is_aromatic']])
                 for atom in replacements) == (ATOM_SOURCE_CODES['explicit'],) * 2
    assert tuple(int(mol._atom_table._sources[atom.idx, FIELD_INDEX['is_aromatic']])
                 for atom in mol.atoms[:-2]) == (ATOM_SOURCE_CODES['rdkit'],) * 4
    mol.calc_implicit_hydrogens()
    assert all(atom.is_aromatic for atom in mol.atoms)
    assert all(bond.is_aromatic for bond in mol.bonds)
    assert tuple(len(atom.heavy_neighbours) for atom in mol.atoms) == (2,) * 6
    assert tuple(_count(atom) for atom in mol.atoms) == (1,) * 6
    assert tuple(_site_facts(atom)[0].source_hydrogens for atom in replacements) == source
    _assert_product_sites(tuple(replacements), (1, 1))
    assert _state(fragment) == fragment_before
