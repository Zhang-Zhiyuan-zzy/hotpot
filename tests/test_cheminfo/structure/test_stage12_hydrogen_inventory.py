"""Hand-counted target/residual inventories and atomic H graph transitions."""
from dataclasses import replace
import pickle

import numpy as np
import pytest

from hotpot import Atom, Molecule, Searcher, Substructure
from hotpot.cheminfo.operations import chemistry
from hotpot.cheminfo.operations.stereochemistry import _gate6_projection
from hotpot.cheminfo.rules.chemistry.bonding.hydrogen import _non_graph_hydrogens
from hotpot.cheminfo.stereo import (
    AtomTransportRecord, HydrogenTransportState, StereoAnchor, StereoAnchorKind,
    StereoCarrier, StereoCarrierKind, StereoPayload, StereoRecord,
    get_stereo_payload, set_stereo_payload,
)
from hotpot.cheminfo.structure._commit import _commit_atom_facts
from hotpot.cheminfo.structure._fields import ATOM_SOURCE_CODES, FIELD_INDEX
from hotpot.cheminfo.structure._identity import _capture_source
from hotpot.cheminfo.structure._mutation import _normalize_atom_fields
from tests.test_cheminfo._stage5_fixtures import factual_state


def _carbon(graph_h=0, target=4, producer='explicit'):
    mol = Molecule()
    carbon = mol.create_atom(atomic_number=6, implicit_hydrogens=target)
    for _ in range(graph_h):
        carbon.add_atom(1)
    _commit_atom_facts(mol, _normalize_atom_fields(
        _capture_source(mol), (0,), (carbon._uid,), {'implicit_hydrogens': (target,)},
        producer=ATOM_SOURCE_CODES[producer]))
    return mol


def _transport(mol, row, current, *, isotope=13):
    payload = get_stereo_payload(mol)
    history = HydrogenTransportState(17, 7, True)
    fact = AtomTransportRecord(row, isotope, 0, current, history, 'test')
    set_stereo_payload(mol, replace(payload, atom_facts=payload.atom_facts + (fact,)))
    return history


@pytest.mark.parametrize('number,graph,count,expected', [
    (6, 0, 4, 4), (6, 1, 4, 3), (6, 4, 4, 0), (6, 4, 2, 0),
    (1, 0, 0, 0), (1, 1, 1, 0), (30, 1, 2, 2), (30, 1, 0, 0),
    (30, 0, -2, 0), (6, 1, None, None),
])
def test_non_graph_projection_has_explicit_heavy_H_and_metal_scopes(number, graph, count, expected):
    assert _non_graph_hydrogens(number, graph, count) == expected


@pytest.mark.parametrize('producer', ['explicit', 'legacy_array', 'rdkit', 'openbabel', 'mca_dto'])
@pytest.mark.parametrize('graph_h', [0, 1, 4])
def test_every_supplied_ordinary_target_materializes_exact_residual_and_retains_source(producer, graph_h):
    mol = _carbon(graph_h, producer=producer)
    carbon = mol.atoms[0]
    history = _transport(mol, 0, HydrogenTransportState(999, min(1, 4-graph_h), False))
    source = pickle.dumps(history)
    assert chemistry._require_non_graph_hydrogens(mol, rows=(0,)) == (4-graph_h,)
    for _ in range(2):
        mol.add_hydrogens(rng=np.random.default_rng(42))
        assert carbon.explicit_hydrogens == 4
        assert carbon.implicit_hydrogens == 4
        assert chemistry._atom_non_graph_hydrogens(carbon) == 0
        fact = get_stereo_payload(mol).atom_facts[0]
        assert fact.current_hydrogens == HydrogenTransportState(0, 0, False)
        assert pickle.dumps(fact.source_hydrogens) == source
        assert fact.isotope == 13 and fact.radical_electrons == 0
        assert mol._atom_table._sources[0, FIELD_INDEX['implicit_hydrogens']] == ATOM_SOURCE_CODES[producer]


def test_residual_selection_preserves_requested_order_and_supplied_target_below_graph():
    mol = _carbon(3, target=2)
    oxygen = mol.create_atom(atomic_number=8, implicit_hydrogens=2)
    assert chemistry._require_non_graph_hydrogens(mol, rows=(oxygen.idx, 0, oxygen.idx)) == (2, 0, 2)
    graph = tuple(mol.atoms), tuple(mol.bonds)
    assert mol.atoms[0].add_hydrogen() == []
    assert (tuple(mol.atoms), tuple(mol.bonds)) == graph
    assert mol.atoms[0].implicit_hydrogens == 2


@pytest.mark.parametrize('entry', ['atom', 'molecule', 'stereo', 'query'])
def test_actual_bracket_conflict_is_a_physical_boundary_without_rewriting_scalar(entry):
    mol = _carbon(target=2)
    _transport(mol, 0, HydrogenTransportState(0, 4, True))
    assert mol.atoms[0].implicit_hydrogens == 2
    before = factual_state(mol)
    actions = {'atom': lambda: mol.atoms[0].add_hydrogen(),
               'molecule': mol.add_hydrogens,
               'stereo': lambda: _gate6_projection(mol),
               'query': lambda: Substructure.from_smarts('[#6;H2]').query_atoms[0].match(mol.atoms[0])}
    with pytest.raises(ValueError, match=r'HYDROGEN.DECLARATION_CONFLICT atom_index=0 T=2 E=0 J=4'):
        actions[entry]()
    assert factual_state(mol) == before
    assert mol.atoms[0].implicit_hydrogens == 2


def test_later_row_declaration_conflict_is_validated_before_the_first_graph_write(monkeypatch):
    mol = _carbon()
    mol.create_atom(atomic_number=6, implicit_hydrogens=2)
    _transport(mol, 1, HydrogenTransportState(123, 4, True))
    before = factual_state(mol)
    def forbidden(*args, **kwargs):
        raise AssertionError('first row graph write preceded whole-selection validation')
    monkeypatch.setattr(Atom, '_add_atom', forbidden)
    with pytest.raises(ValueError, match=r'atom_index=1 T=2 E=0 J=4'):
        mol.add_hydrogens()
    assert factual_state(mol) == before


def test_explicit_num_consumes_bracket_first_then_implicit_and_retains_target_history():
    mol = _carbon()
    history = _transport(mol, 0, HydrogenTransportState(2, 2, False))
    carbon = mol.atoms[0]
    assert len(carbon.add_hydrogen(num=1)) == 1
    fact = get_stereo_payload(mol).atom_facts[0]
    assert fact.current_hydrogens == HydrogenTransportState(2, 1, False)
    assert carbon.implicit_hydrogens == 4 and carbon.explicit_hydrogens == 1
    assert len(carbon.add_hydrogen(num=2)) == 2
    fact = get_stereo_payload(mol).atom_facts[0]
    assert fact.current_hydrogens == HydrogenTransportState(1, 0, False)
    assert fact.source_hydrogens == history
    assert carbon.implicit_hydrogens == 4 and carbon.explicit_hydrogens == 3
    assert len(carbon.add_hydrogen()) == 1
    assert get_stereo_payload(mol).atom_facts[0].current_hydrogens == HydrogenTransportState(0, 0, False)
    assert carbon.add_hydrogen() == []


def test_num0_is_a_factually_exact_noop_even_with_a_bracket_conflict():
    mol = _carbon(target=2)
    _transport(mol, 0, HydrogenTransportState(999, 4, True))
    before = factual_state(mol)
    assert mol.atoms[0].add_hydrogen(num=0) == []
    assert factual_state(mol) == before


@pytest.mark.parametrize('graph_h', [0, 1, 4])
def test_SMARTS_python_native_and_gate6_share_the_hand_counted_methane_inventory(graph_h):
    mol = _carbon(graph_h)
    query = Substructure.from_smarts('[#6;H4;X4;v4]')
    assert query.query_atoms[0].match(mol.atoms[0])
    assert len(Searcher(query).search(mol)) == 1
    graph, issues = _gate6_projection(mol)
    assert issues == ()
    assert len(graph.virtual_hydrogens[0]) == 4-graph_h
    assert sum(len(h) for h in graph.virtual_hydrogens) == 4-graph_h
    assert mol.sum_implicit_hydrogens == 4


def _tetrahedral_virtual_H():
    mol = Molecule()
    center = mol.create_atom(atomic_number=6, implicit_hydrogens=1)
    for number in [9, 17, 35]:
        center.add_atom(number)
    record = StereoRecord(StereoAnchor(StereoAnchorKind.ATOM, (0,)),
        (StereoCarrier(StereoCarrierKind.ATOM, 1), StereoCarrier(StereoCarrierKind.IMPLICIT_H),
         StereoCarrier(StereoCarrierKind.ATOM, 3), StereoCarrier(StereoCarrierKind.ATOM, 2)),
        'TH1', None, 'test', 'TH1')
    set_stereo_payload(mol, StereoPayload(1, (record,), (), ()))
    _transport(mol, 0, HydrogenTransportState(0, 1, True))
    return mol


@pytest.mark.parametrize('explicit_num', [None, 1])
def test_materialization_maps_virtual_H_in_its_original_ordered_carrier_position(explicit_num):
    mol = _tetrahedral_virtual_H()
    old = get_stereo_payload(mol)
    added = mol.atoms[0].add_hydrogen(num=explicit_num)
    assert len(added) == 1
    new = get_stereo_payload(mol)
    assert new.records[0] == replace(old.records[0], carriers=(old.records[0].carriers[0],
        StereoCarrier(StereoCarrierKind.ATOM, added[0].idx), *old.records[0].carriers[2:]))
    assert new.atom_facts[0].source_hydrogens == old.atom_facts[0].source_hydrogens
    assert new.atom_facts[0].current_hydrogens == HydrogenTransportState(0, 0, True)


def test_late_materialization_failure_restores_ordered_carriers_and_current_source_state(monkeypatch):
    mol = _tetrahedral_virtual_H()
    before = factual_state(mol)
    materialize = chemistry._materialize_current_hydrogens
    def fail(atom):
        materialize(atom)
        assert get_stereo_payload(mol).records[0].carriers[1].kind is StereoCarrierKind.ATOM
        raise RuntimeError('late current-H failure')
    monkeypatch.setattr(chemistry, '_materialize_current_hydrogens', fail)
    with pytest.raises(RuntimeError, match='late current-H failure'):
        mol.add_hydrogens(rng=np.random.default_rng(42))
    assert factual_state(mol) == before


@pytest.mark.parametrize('kind', [StereoAnchorKind.BOND, StereoAnchorKind.AXIS])
def test_endpoint_virtual_carriers_materialize_in_their_own_ordered_groups(kind):
    mol = Molecule()
    endpoint_count = 2 if kind is StereoAnchorKind.BOND else 3
    for row in range(endpoint_count):
        mol.create_atom(atomic_number=6, implicit_hydrogens=0 if endpoint_count == 3 and row == 1 else 1)
    for row in range(endpoint_count-1):
        mol.add_bond(row, row+1, 2)
    first = mol.atoms[0]
    last = mol.atoms[endpoint_count-1]
    first.add_atom(9)
    last.add_atom(17)
    record = StereoRecord(StereoAnchor(kind, tuple(range(endpoint_count))),
        (StereoCarrier(StereoCarrierKind.ATOM, endpoint_count), StereoCarrier(StereoCarrierKind.IMPLICIT_H),
         StereoCarrier(StereoCarrierKind.IMPLICIT_H), StereoCarrier(StereoCarrierKind.ATOM, endpoint_count+1)),
        'DB1' if kind is StereoAnchorKind.BOND else 'AX1', None, 'test', 'ordered')
    set_stereo_payload(mol, StereoPayload(1, (record,), (), ()))
    first_h, = first.add_hydrogen(num=1)
    halfway = get_stereo_payload(mol).records[0]
    assert halfway.carriers[1] == StereoCarrier(StereoCarrierKind.ATOM, first_h.idx)
    assert halfway.carriers[2] == record.carriers[2]
    last_h, = last.add_hydrogen()
    completed = get_stereo_payload(mol).records[0]
    assert completed == replace(record, carriers=(record.carriers[0],
        StereoCarrier(StereoCarrierKind.ATOM, first_h.idx),
        StereoCarrier(StereoCarrierKind.ATOM, last_h.idx), record.carriers[3]))


def test_an_unrelated_opaque_record_is_preserved_during_graph_H_materialization():
    mol = _carbon()
    opaque = StereoRecord(StereoAnchor(StereoAnchorKind.ATOM, ()), (),
                          'OPAQUE', None, 'unregistered', 'source-only')
    set_stereo_payload(mol, StereoPayload(1, (opaque,), (), ()))
    mol.add_hydrogens(rng=np.random.default_rng(42))
    assert get_stereo_payload(mol).records == (opaque,)


@pytest.mark.parametrize('graph_h', [0, 1, 4])
@pytest.mark.parametrize('origin', ['inferred', 'rdkit', 'openbabel'])
def test_neutral_methane_default_charge_completes_only_its_working_copy(origin, graph_h):
    if origin == 'inferred':
        mol = Molecule()
        carbon = mol.create_atom(atomic_number=6)
        for _ in range(graph_h):
            carbon.add_atom(1)
    elif origin == 'rdkit':
        from rdkit import Chem
        from hotpot.cheminfo.operations.conversion import from_rdmol
        graph = Chem.RWMol()
        graph.AddAtom(Chem.Atom(6))
        for row in range(graph_h):
            hydrogen = Chem.Atom(1)
            hydrogen.SetNoImplicit(True)
            graph.AddAtom(hydrogen)
            graph.AddBond(0, row+1, Chem.BondType.SINGLE)
        graph = graph.GetMol()
        Chem.SanitizeMol(graph)
        mol = from_rdmol(graph, Molecule())
    else:
        from hotpot.cheminfo.adapters.openbabel._runtime import ob
        from hotpot.cheminfo.operations.conversion import obmol2mol
        graph = ob.OBMol()
        graph.BeginModify()
        carbon = graph.NewAtom()
        carbon.SetAtomicNum(6)
        carbon.SetImplicitHCount(4-graph_h)
        for row in range(graph_h):
            hydrogen = graph.NewAtom()
            hydrogen.SetAtomicNum(1)
            hydrogen.SetImplicitHCount(0)
            graph.AddBond(1, row+2, 1)
        graph.EndModify()
        mol = obmol2mol(graph, Molecule())
    # Imported toolkit charge is deliberately removed through its canonical
    # writer, so a declaration cannot mask the actual default calculation.
    from hotpot.cheminfo.structure._commit import _commit_molecular_declarations
    _commit_molecular_declarations(mol, _capture_source(mol), charge=(False, 0, None))
    chemistry._require_implicit_hydrogens(mol)
    assert not mol._charge_present
    assert mol.sum_implicit_hydrogens == 4
    assert mol.atoms[0].explicit_hydrogens == graph_h
    before = factual_state(mol)
    graph_members = tuple(mol.atoms), tuple(mol.bonds)
    source_transport = pickle.dumps(get_stereo_payload(mol))
    assert mol.calc_mol_default_charge() == 0
    assert mol.charge == 0 and not mol._charge_present
    assert mol.sum_implicit_hydrogens == 4
    assert (tuple(mol.atoms), tuple(mol.bonds)) == graph_members
    assert pickle.dumps(get_stereo_payload(mol)) == source_transport
    assert factual_state(mol) == before


def test_raw_target_sum_remains_available_when_a_physical_bracket_projection_conflicts():
    mol = _carbon(target=2)
    _transport(mol, 0, HydrogenTransportState(999, 4, True))
    before = factual_state(mol)
    assert mol.sum_implicit_hydrogens == 2
    assert factual_state(mol) == before
    with pytest.raises(ValueError, match='HYDROGEN.DECLARATION_CONFLICT'):
        chemistry._require_non_graph_hydrogens(mol)
