"""Real aromatic preparation across Crystal generation and editable PyG transport."""

import builtins
from dataclasses import fields
import math

import numpy as np
import pytest

from hotpot import Crystal, Molecule, SpaceGroup
from hotpot.cheminfo.operations import chemistry
from hotpot.cheminfo.operations._aromatic_protocol import _evaluate_aromatic_assignment
from hotpot.cheminfo.rules.chemistry.bonding import kekule
from hotpot.cheminfo.structure._fields import ATOM_SOURCE_CODES, FIELD_INDEX, FIELD_PRESENCE_BITS
from hotpot.cheminfo.structure._results import _import_partial_charges

torch = pytest.importorskip('torch')
from torch_geometric.data import Batch

BASIS = np.array(((0., 8., 0.), (-9., 0., 0.), (0., 0., 10.)))
ZERO = (0, 0, 0)
NODE_ORDER = (2, 0, 5, 1, 4, 3)
EDGE_ORDER = (4, 1, 5, 0, 3, 2)
OCCUPANCIES = (.5, None, 0., 1., None, .75)
CHARGES = (0., None, -.1, .1, 0., 0.)
NODE_FIELDS = ('atom_key', 'z', 'formal_charge', 'total_h', 'aromatic', 'atom_id', 'pos',
               'atom_present', 'atom_source', 'representation_available', 'atom_constraint',
               'occupancy', 'occupancy_present')
EDGE_FIELDS = ('bond_key', 'bond_order', 'bond_kind', 'bond_direction', 'bond_aromatic',
               'bond_id', 'bond_constraint', 'bond_image', 'bond_image_present')


@pytest.fixture(autouse=True)
def prohibit_toolkit_routes(monkeypatch):
    original = builtins.__import__

    def guarded(name, *args, **kwargs):
        if name.split('.')[0] in ('rdkit', 'openbabel', 'ccdc', 'gemmi', 'pybel'):
            raise AssertionError('Forbidden toolkit in aromatic/Crystal/Data seam: ' + name)
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, '__import__', guarded)


def _observe_existing_rule(monkeypatch):
    assert chemistry._calculate_aromatic_assignment is _evaluate_aromatic_assignment
    assert _evaluate_aromatic_assignment is kekule._calculate_aromatic_assignment
    calls = []

    def observed(snapshot, *, kekule):
        proposal = _evaluate_aromatic_assignment(snapshot, kekule=kekule)
        assert proposal is not None and proposal.source is snapshot.source
        calls.append((kekule, len(snapshot.atom_uids)))
        return proposal

    monkeypatch.setattr(chemistry, '_calculate_aromatic_assignment', observed)
    return calls


def _benzene():
    points = np.array([(-4.5 + 1.4 * math.cos(row * math.pi / 3),
                        4. + 1.4 * math.sin(row * math.pi / 3), 5.) for row in range(6)])
    mol = Molecule()
    for row, point in enumerate(points):
        mol.create_atom(atomic_number=6, formal_charge=0, implicit_hydrogens=1,
                        coordinates=point, id=100 + row)
    for row in range(6):
        mol.add_bond(row, (row + 1) % 6, bond_order=2. if row % 2 == 0 else 1.)
    assert tuple(bond.bond_order for bond in mol.bonds) == (2., 1., 2., 1., 2., 1.)
    np.testing.assert_allclose(np.linalg.norm(points - np.roll(points, -1, axis=0), axis=1),
                               np.full(6, 1.4), rtol=0, atol=1e-12)
    assert mol.determine_rings_aromatic() is None
    return mol, points


def _cell(mol, images):
    return Crystal.from_matrix(BASIS, molecule=mol, space_group=SpaceGroup(1),
        bond_images=images, site_relations={'equivalent_images': (), 'occupancies': OCCUPANCIES,
            'disorder': tuple(('fixture', row) for row in range(6))})


def _raw(mol):
    return mol._capture_factual_record(include_frames=False)[0]


def _state(mol):
    record, metadata = mol._capture_factual_record(include_frames=False)
    return tuple((field.name, getattr(record, field.name)) for field in fields(record)), metadata


def _assert_benzene_facts(data, rings):
    count = 6 * rings
    assert data.z.tolist() == [6] * count
    assert data.formal_charge.tolist() == [0] * count
    assert data.total_h.tolist() == [1] * count
    assert data.atom_present[:, :3].all() and data.atom_present[:, 5:].all()
    assert data.atom_source[:, 2].tolist() == [ATOM_SOURCE_CODES['explicit']] * count
    assert data.aromatic.tolist() == [True] * count
    assert data.bond_aromatic.tolist() == [True] * count
    assert sorted(data.bond_order.tolist()) == [1.] * (3 * rings) + [2.] * (3 * rings)
    assert sorted(data.bond_kind.tolist()) == [2] * (3 * rings) + [3] * (3 * rings)
    adjacency = [[] for _ in range(count)]
    for (first, second), order in zip(data.edge_index.T.tolist(), data.bond_order.tolist()):
        adjacency[first].append(order)
        adjacency[second].append(order)
    assert all(sorted(orders) == [1., 2.] for orders in adjacency)
    assert not data.net_charge_present[0] and not data.spin_mult_present[0]


def test_real_aromatic_assignment_transport_expansion_and_current_preparation(monkeypatch):
    calls = _observe_existing_rule(monkeypatch)
    mol, points = _benzene()
    assert calls == [(True, 6)]
    _import_partial_charges(mol, CHARGES, producer=ATOM_SOURCE_CODES['explicit'],
                            method='literal-supplied-seam-fixture')
    source = _cell(mol, (ZERO,) * 6)
    source_before = _state(mol)
    data = source.to_pyg()
    _assert_benzene_facts(data, 1)
    source_orders = data.bond_order.clone()
    source_kinds = data.bond_kind.clone()
    order = torch.tensor(NODE_ORDER)
    inverse = torch.argsort(order)
    for name in NODE_FIELDS:
        data[name] = data[name][order]
    for name in ('edge_index', 'angle_index', 'torsion_index', 'wiberg_index'):
        data[name] = inverse[data[name]]
    edge_order = torch.tensor(EDGE_ORDER)
    data.edge_index = data.edge_index[:, edge_order]
    for name in EDGE_FIELDS:
        data[name] = data[name][edge_order]
    unknown = Molecule()
    unknown.create_atom(atomic_number=8)
    batch = Batch.from_data_list([Molecule().to_pyg(), data, unknown.to_pyg()]).cpu()
    assert batch.ptr.tolist() == [0, 0, 6, 7]
    parts = batch.to_data_list()
    restored = Crystal.from_pyg(parts[1])
    absent = Molecule.from_pyg(parts[2]).to_pyg()
    assert not absent.atom_present[0, 1:4].any()
    assert not absent.atom_present[0, 5:].any()
    assert calls == [(True, 6)]
    transported = restored.to_pyg()
    _assert_benzene_facts(transported, 1)
    assert torch.equal(transported.bond_order, source_orders[edge_order])
    assert torch.equal(transported.bond_kind, source_kinds[edge_order])
    assert transported.atom_id.tolist() == [100 + row for row in NODE_ORDER]
    np.testing.assert_array_equal(restored.matrix, BASIS)
    np.testing.assert_allclose(restored.molecule.coordinates, points[list(NODE_ORDER)], rtol=0, atol=1e-12)
    assert restored.bond_images == (ZERO,) * 6
    assert restored.site_relations['occupancies'] == tuple(OCCUPANCIES[row] for row in NODE_ORDER)
    history = restored.molecule.properties['pyg_observations']
    assert history['initial_atom_keys'] == list(NODE_ORDER)
    observation = history['observations']['current_source'][0]
    assert observation['kind'] == 'partial_charge'
    assert observation['values'] == list(CHARGES)
    assert observation['subjects']['atom_keys'] == list(range(6))
    assert restored.molecule._partial_charge_result is None
    masks = np.frombuffer(_raw(restored.molecule).atom_presence, dtype='<u4')
    assert not (masks & FIELD_PRESENCE_BITS[FIELD_INDEX['partial_charge']]).any()

    generated = restored.expand((2, 1, 1))
    assert calls == [(True, 6)]
    assert generated.representation == 'supercell' and generated.space_group == SpaceGroup(1)
    np.testing.assert_array_equal(generated.matrix, BASIS * np.array([[2.], [1.], [1.]]))
    expected_points = np.array([points[row] + shift * BASIS[0]
                                for row in NODE_ORDER for shift in range(2)])
    np.testing.assert_allclose(generated.molecule.coordinates, expected_points, rtol=0, atol=1e-10)
    assert generated.bond_images == (ZERO,) * 12
    assert generated.site_relations['occupancies'] == tuple(OCCUPANCIES[row]
        for row in NODE_ORDER for _ in range(2))
    for destination, origins in enumerate(generated.image_map):
        assert origins == ((destination // 2, SpaceGroup(1).identity(), (destination % 2, 0, 0)),)
    assert generated.molecule._partial_charge_result is None
    assert generated.molecule.properties['pyg_observations'] == history
    projected = generated.to_pyg()
    _assert_benzene_facts(projected, 2)
    assert projected.hotpot_observations[0]['current_source'] == []
    assert projected.hotpot_observations[0]['historical'] is not None

    assert generated.molecule.calc_atom_valence(assign_aromatic=True) is None
    assert len(calls) > 1 and all(call == (False, 12) for call in calls[1:])
    prepared_calls = len(calls)
    assert all(atom.is_aromatic for atom in generated.molecule.atoms)
    assert all(bond.is_aromatic for bond in generated.molecule.bonds)
    assert [atom.implicit_hydrogens for atom in generated.molecule.atoms] == [1] * 12
    assert len(calls) == prepared_calls
    _assert_benzene_facts(generated.to_pyg(), 2)
    assert _state(mol) == source_before


def test_unknown_periodic_images_survive_transport_and_reject_replication(monkeypatch):
    calls = _observe_existing_rule(monkeypatch)
    mol, _ = _benzene()
    source = _cell(mol, (None,) * 6)
    source_before = _state(mol)
    data = source.to_pyg()
    assert not data.bond_image_present.any()
    data.bond_image[:] = torch.tensor((9, -8, 7))
    part = Batch.from_data_list([Molecule().to_pyg(), data]).to_data_list()[1]
    restored = Crystal.from_pyg(part)
    assert restored.bond_images == (None,) * 6
    assert restored.site_relations['occupancies'] == OCCUPANCIES
    assert calls == [(True, 6)]
    before = _state(restored.molecule)
    with pytest.raises(ValueError, match='CRYSTAL-UNSUPPORTED-PROJECTION: expansion target cell needs known Bond images'):
        restored.expand((2, 1, 1))
    assert _state(restored.molecule) == before
    assert _state(mol) == source_before
    assert calls == [(True, 6)]
    assert restored.molecule.calc_atom_valence(assign_aromatic=True) is None
    assert len(calls) > 1 and all(call == (False, 6) for call in calls[1:])
    result = restored.to_pyg()
    _assert_benzene_facts(result, 1)
    assert not result.bond_image_present.any()
    assert result.occupancy_present.tolist() == [True, False, True, True, False, True]
