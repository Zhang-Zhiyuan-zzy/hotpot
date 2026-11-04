"""Independent Crystal/Data joins, directed images, history and isolation."""

from copy import deepcopy

import numpy as np
import pytest

from hotpot import BondKind, Crystal, Molecule, SpaceGroup
from hotpot.cheminfo.structure._crystal import _read_crystal_record

torch = pytest.importorskip('torch')
from torch_geometric.data import Batch, Data

ZERO = (0, 0, 0)
NODE_FIELDS = ('atom_key', 'z', 'formal_charge', 'total_h', 'aromatic', 'atom_id', 'pos',
               'atom_present', 'atom_source', 'representation_available', 'atom_constraint',
               'occupancy', 'occupancy_present')
EDGE_FIELDS = ('bond_key', 'bond_order', 'bond_kind', 'bond_direction', 'bond_aromatic',
               'bond_id', 'bond_constraint', 'bond_image', 'bond_image_present')


def crystal(*, generated=False):
    mol = Molecule()
    for number, point in ((6, (0., 0., 0.)), (8, (1., 0., 0.)), (7, (2., 1., 0.))):
        mol.create_atom(atomic_number=number, coordinates=point, formal_charge=0,
                        implicit_hydrogens=0)
    mol.add_bond(0, 1, 1., bond_kind=BondKind.DATIVE, bond_direction='atom1_to_atom2')
    mol.add_bond(1, 2, 2.)
    value = Crystal(7, 8, 9, 90, 90, 90, molecule=mol, space_group=SpaceGroup(1),
        bond_images=(ZERO, ZERO), site_relations={'equivalent_images': (),
            'occupancies': (.2, None, 0.), 'disorder': (('A', 1), None, ('B', 2))})
    return value.expand((2, 1, 1)) if generated else value


@pytest.mark.parametrize('generated', (False, True))
def test_directed_reversal_with_inverse_image_and_direction(generated):
    value = crystal(generated=generated)
    data = value.to_pyg()
    data.bond_image[0] = torch.tensor((1, -2, 3))
    data.bond_image_present[0] = True
    data.edge_index[:, 0] = data.edge_index.flip(0)[:, 0]
    data.bond_image[0] *= -1
    data.bond_direction[0] *= -1
    other = Crystal.from_pyg(data)
    assert other.bond_images[0] == (-1, 2, -3)
    assert (other.molecule.bonds[0].a1idx, other.molecule.bonds[0].a2idx) == tuple(data.edge_index[:, 0].tolist())
    assert other.molecule.bonds[0].bond_direction == 'atom2_to_atom1'
    assert value.bond_images[0] == ZERO
    if generated:
        assert _read_crystal_record(other)[6:] == _read_crystal_record(value)[6:]


@pytest.mark.parametrize('generated', (False, True))
def test_noncontiguous_keys_and_current_references_survive_reordering(generated):
    value = crystal(generated=generated)
    group = value.space_group
    value.set_site_relations({'equivalent_images': (((0, group.identity(), ZERO),
        (2, group.identity(), (1, 0, 0))),), 'occupancies': value.site_relations['occupancies'],
        'disorder': value.site_relations['disorder']})
    before = _read_crystal_record(value)
    data = value.to_pyg()
    new_keys = [31 + 7 * i for i in range(data.num_nodes)]
    data.atom_key = torch.tensor(new_keys)
    context = data.hotpot_crystal[0]
    context['current_atom_keys'] = new_keys.copy()
    for entry in context['relations']['equivalent_images']:
        for member in entry:
            member[0] = new_keys[member[0]]
    new_bonds = [71 + 11 * i for i in range(data.bond_key.numel())]
    data.bond_key = torch.tensor(new_bonds)
    context['current_bond_keys'] = new_bonds.copy()
    for item, key in zip(data.hotpot_bonds[0], new_bonds):
        item['bond_key'] = key
    order = torch.arange(data.num_nodes - 1, -1, -1)
    inverse = torch.argsort(order)
    for name in NODE_FIELDS:
        data[name] = data[name][order]
    for name in ('edge_index', 'angle_index', 'torsion_index', 'wiberg_index'):
        data[name] = inverse[data[name]]
    edge_order = torch.arange(data.bond_key.numel() - 1, -1, -1)
    data.edge_index = data.edge_index[:, edge_order]
    for name in EDGE_FIELDS:
        data[name] = data[name][edge_order]
    after = _read_crystal_record(Crystal.from_pyg(data))
    assert after[3][0] == tuple(tuple((int(inverse[row]), operation, image)
        for row, operation, image in entry) for entry in before[3][0])
    assert after[6] == before[6]
    assert after[7] == (None if before[7] is None else tuple(before[7][row] for row in order.tolist()))
    assert after[8] == (None if before[8] is None else tuple(before[8][row] for row in edge_order.tolist()))


def test_stale_generated_basis_map_rejects_and_explicit_unknown_basis_imports():
    value = crystal(generated=True)
    data = value.to_pyg()
    data.cell_matrix *= 2
    with pytest.raises(ValueError, match='CRYSTAL-IMAGE-MAP-INTEGRITY: exact historical basis map'):
        Crystal.from_pyg(data)
    historical = deepcopy(data.hotpot_crystal[0]['origin_binding'])
    data.hotpot_crystal[0]['origin_binding']['basis_map'] = None
    other = Crystal.from_pyg(data)
    np.testing.assert_array_equal(other.matrix, value.matrix * 2)
    assert _read_crystal_record(other)[6][4] is None
    assert other.to_pyg().hotpot_crystal[0]['origin_binding']['sites'] == historical['sites']


def test_mixed_batch_ragged_operations_and_result_buffers_are_independent():
    a = crystal(generated=True)
    b = Crystal(7, 8, 9, 90, 90, 90, space_group=SpaceGroup(19))
    c = Crystal(7, 8, 9, 90, 90, 90, molecule=Molecule())
    graphs = [a.to_pyg(), b.to_pyg(), c.to_pyg(), Molecule().to_pyg()]
    batch = Batch.from_data_list(graphs).cpu()
    assert batch.num_graphs == 4
    parts = batch.to_data_list()
    assert [part.space_group_R.shape[0] for part in parts] == [1, 4, 0, 0]
    restored = [Crystal.from_pyg(part) for part in parts[:3]]
    assert restored[1].molecule is None
    assert restored[2].molecule is not None and len(restored[2].molecule.atoms) == 0
    before = _read_crystal_record(restored[0])
    batch.cell_matrix[0, 0, 0] = 999
    batch.occupancy[0] = .99
    batch.bond_image[0, 0] = 42
    batch.hotpot_crystal[0][0]['origin_binding']['sites'][0][0] = 999
    assert _read_crystal_record(restored[0]) == before
    assert _read_crystal_record(a) == before
    with pytest.raises(ValueError, match='pyg_mapping Batch'):
        Crystal.from_pyg(batch)


def test_foreign_paired_unknown_image_bits_ignore_non_authoritative_payload():
    data = Data(num_nodes=2, z=torch.tensor([6, 8]), edge_index=torch.tensor([[0, 1], [1, 0]]),
        basis=torch.eye(3).reshape(1, 3, 3), images=torch.tensor([[8, 9, 10], [2, 3, 4]]),
        known=torch.tensor([False, False]))
    mapping = {'fields': {'z': 'z', 'edge_index': 'edge_index', 'cell_matrix': 'basis',
        'bond_image': 'images', 'bond_image_present': 'known'},
        'edge_mode': 'paired', 'coordinate_unit': 'angstrom'}
    assert Crystal.from_pyg(data, field_map=mapping).bond_images == (None,)
    data.known[0] = True
    with pytest.raises(ValueError, match='pyg_mapping paired bond_image_present'):
        Crystal.from_pyg(data, field_map=mapping)


@pytest.mark.parametrize('path', [
    ('current_atom_keys',), ('current_bond_keys',), ('relations',), ('image_map',),
    ('bond_origins',), ('relations', 'disorder'), ('relations', 'equivalent_images'),
])
def test_missing_required_context_join_is_named_local_error(path):
    data = crystal().to_pyg()
    container = data.hotpot_crystal[0]
    for key in path[:-1]:
        container = container[key]
    del container[path[-1]]
    with pytest.raises(ValueError, match=rf'pyg_schema hotpot_crystal.*{path[-1]}'):
        Crystal.from_pyg(data)


def test_missing_current_equivalence_member_is_named_local_error():
    value = crystal()
    group = value.space_group
    value.set_site_relations({'equivalent_images': (((0, group.identity(), ZERO),
        (1, group.identity(), ZERO)),), 'occupancies': (.2, None, 0.),
        'disorder': (('A', 1), None, ('B', 2))})
    data = value.to_pyg()
    data.hotpot_crystal[0]['relations']['equivalent_images'][0][0][0] = 999
    with pytest.raises(ValueError, match='pyg_mapping crystal.equivalent_images.*999'):
        Crystal.from_pyg(data)


@pytest.mark.parametrize('declaration', ('net_charge', 'spin_mult', 'identifier'))
def test_standalone_cannot_discard_edited_molecular_declaration(declaration):
    data = Crystal(7, 8, 9, 90, 90, 90).to_pyg()
    if declaration == 'identifier':
        data.hotpot_graph[0]['identifier']['items'] = [True, 'declared']
    else:
        data[declaration][0] = 2.
        data[declaration + '_present'][0] = True
    with pytest.raises(ValueError, match='pyg_mapping standalone Crystal'):
        Crystal.from_pyg(data)


def test_standalone_ignores_numeric_declaration_payload_with_false_presence():
    data = Crystal(7, 8, 9, 90, 90, 90).to_pyg()
    data.net_charge[0] = 2.
    data.spin_mult[0] = 3.
    assert not data.net_charge_present[0] and not data.spin_mult_present[0]
    restored = Crystal.from_pyg(data)
    assert restored.molecule is None
    exported = restored.to_pyg()
    assert not exported.net_charge_present[0] and not exported.spin_mult_present[0]
