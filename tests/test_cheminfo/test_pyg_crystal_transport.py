"""Independent periodic facts, current edits and mixed ordinary Data workflows."""

import builtins
from copy import deepcopy
import io

import numpy as np
import pytest

from hotpot import Crystal, Molecule, SpaceGroup
from hotpot.cheminfo.structure._crystal import _read_crystal_record
from hotpot.cheminfo.structure._crystal_codec import _crystal_to_data, _group_to_data

torch = pytest.importorskip('torch')
from torch_geometric.data import Batch, Data
from torch_geometric.loader import DataLoader

ZERO = (0, 0, 0)
NODE_FIELDS = ('atom_key', 'z', 'formal_charge', 'total_h', 'aromatic', 'atom_id', 'pos',
               'atom_present', 'atom_source', 'representation_available', 'atom_constraint',
               'occupancy', 'occupancy_present')
EDGE_FIELDS = ('bond_key', 'bond_order', 'bond_kind', 'bond_direction', 'bond_aromatic',
               'bond_id', 'bond_constraint', 'bond_image', 'bond_image_present')


def molecule(points=((1., 0., 0.), (2., 1., 0.), (3., 1., 0.))):
    value = Molecule()
    for i, point in enumerate(points):
        value.create_atom(atomic_number=(6, 8, 7)[i % 3], formal_charge=0,
                          implicit_hydrogens=0, coordinates=point)
    for row in range(len(points) - 1):
        value.add_bond(row, row + 1, bond_order=1)
    return value


def crystal(*, group=None, images=((1, -2, 0), None)):
    group = SpaceGroup(2) if group is None else group
    relations = {'equivalent_images': (((0, group.identity(), ZERO),
                                        (1, group.lift(1), (2, 0, -1))),),
                 'occupancies': (.5, 0., None), 'disorder': (('A', 1), ('A', 2), None)}
    return Crystal.from_matrix(((0., 5., 0.), (-6., 0., 0.), (0., 0., 7.)), molecule=molecule(),
                               space_group=group, site_relations=relations, bond_images=images)


def state(value):
    if value.molecule is None:
        return _read_crystal_record(value), None
    record, metadata = value.molecule._capture_factual_record(include_frames=False)
    return record.cell, (record.atom_values, record.atom_presence, record.atom_sources,
                         record.bond_endpoints, record.bond_values, metadata)


@pytest.fixture(autouse=True)
def prohibit_toolkits(monkeypatch):
    original = builtins.__import__
    def guarded(name, *args, **kwargs):
        if name.split('.')[0] in ('rdkit', 'openbabel', 'ccdc', 'gemmi', 'pybel'):
            raise AssertionError('Forbidden toolkit in general Crystal/Data transport: ' + name)
        return original(name, *args, **kwargs)
    monkeypatch.setattr(builtins, '__import__', guarded)


def test_exact_oriented_current_fields_and_fresh_roundtrip():
    value = crystal()
    before = state(value)
    data = value.to_pyg()
    assert type(data) is Data and data.hotpot_schema == ['hotpot.molecule/2']
    assert 'cell_parameters' not in data
    assert data.cell_matrix.dtype == torch.float64 and data.cell_matrix.shape == (1, 3, 3)
    assert data.bond_image.dtype == torch.int64 and data.bond_image.tolist() == [[1, -2, 0], [0, 0, 0]]
    assert data.bond_image_present.tolist() == [True, False]
    assert data.occupancy.tolist() == [.5, 0., 0.] and data.occupancy_present.tolist() == [True, True, False]
    assert data.space_group_R.dtype == torch.int64 and data.space_group_R.shape == (2, 3, 3)
    context = data.hotpot_crystal[0]
    assert 'matrix' not in context and 'bond_images' not in context and 'occupancies' not in context['relations']
    assert context['group']['to_reference'][0] == [['1', '0', '0'], ['0', '1', '0'], ['0', '0', '1']]
    restored = Crystal.from_pyg(data)
    assert restored is not value and restored.molecule is not value.molecule
    assert restored.molecule.crystal is restored
    assert _read_crystal_record(restored) == _read_crystal_record(value)
    np.testing.assert_array_equal(restored.molecule.coordinates, value.molecule.coordinates)
    assert state(value) == before
    data.cell_matrix[0, 0, 1] = 50.
    data.pos[0, 0] = 90.
    assert state(value) == before


def test_actual_tensor_edits_and_unknown_to_explicit_zero():
    value = crystal()
    data = value.to_pyg()
    data.cell_matrix *= 2
    data.pos[0] = torch.tensor((8., 9., 10.))
    data.occupancy[2] = .25
    data.occupancy_present[2] = True
    data.bond_image_present[1] = True
    edited = Crystal.from_pyg(data)
    np.testing.assert_array_equal(edited.matrix, value.matrix * 2)
    np.testing.assert_array_equal(edited.molecule.coordinates[0], (8., 9., 10.))
    assert edited.site_relations['occupancies'] == (.5, 0., .25)
    assert edited.bond_images == ((1, -2, 0), ZERO)
    assert value.bond_images[1] is None


@pytest.mark.parametrize('generated', (False, True))
def test_current_key_join_survives_node_and_edge_order(generated):
    value = crystal(images=(ZERO, ZERO))
    value.set_site_relations({'equivalent_images': (), 'occupancies': (.5, 0., None),
                             'disorder': (('A', 1), ('A', 2), None)})
    if generated:
        value = value.expand((2, 1, 1))
    original = _read_crystal_record(value)
    data = value.to_pyg()
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
    restored = Crystal.from_pyg(data)
    result = _read_crystal_record(restored)
    rows, bonds = order.tolist(), edge_order.tolist()
    assert result[:3] == original[:3]
    assert result[3][1] == tuple(original[3][1][i] for i in rows)
    assert result[3][2] == tuple(original[3][2][i] for i in rows)
    assert result[4] == tuple(original[4][i] for i in bonds)
    assert result[5:7] == original[5:7]
    assert result[7] == (None if original[7] is None else tuple(original[7][i] for i in rows))
    assert result[8] == (None if original[8] is None else tuple(original[8][i] for i in bonds))
    np.testing.assert_array_equal(restored.molecule.coordinates, value.molecule.coordinates[rows])


def test_equivalent_current_rows_remap_but_historical_source_rows_do_not():
    value = crystal()
    data = value.to_pyg()
    order = torch.tensor([2, 0, 1]); inverse = torch.argsort(order)
    for name in NODE_FIELDS:
        data[name] = data[name][order]
    for name in ('edge_index', 'angle_index', 'torsion_index', 'wiberg_index'):
        data[name] = inverse[data[name]]
    other = Crystal.from_pyg(data)
    expected = tuple(tuple((int(inverse[row]), operation, image) for row, operation, image in entry)
                     for entry in value.site_relations['equivalent_images'])
    assert other.site_relations['equivalent_images'] == expected


def test_unknown_group_stays_unknown_with_zero_representatives():
    value = Crystal(5, 6, 7, 90, 90, 90, molecule=molecule(), bond_images=(None, ZERO))
    data = value.to_pyg()
    assert data.space_group_R.shape == (0, 3, 3) and data.space_group_t.shape == (0, 3)
    restored = Crystal.from_pyg(data)
    assert restored.space_group is None and restored.symm_units is None
    assert restored.bond_images == (None, ZERO)


@pytest.mark.parametrize('number', (4, 5, 19, 62))
def test_screw_centring_and_orthorhombic_exact_contexts(number):
    group = SpaceGroup(number)
    value = Crystal(8, 9, 10, 90, 90, 90, molecule=molecule(), space_group=group)
    data = value.to_pyg()
    assert data.space_group_R.shape[0] == group.cell_order
    other = Crystal.from_pyg(data)
    assert other.space_group == group
    assert [operation.xyz for operation in other.symm_units] == [operation.xyz for operation in group.symm_units]


def test_changed_origin_and_fractional_group_terms_stay_exact():
    group = SpaceGroup(2).change_setting(((1, 0, 0), (0, 1, 0), (0, 0, 1)), ('1/7', '-2/9', '3/11'))
    value = Crystal(8, 9, 10, 90, 90, 90, molecule=molecule(), space_group=group)
    data = value.to_pyg()
    other = Crystal.from_pyg(data)
    assert other.space_group == group and other.space_group.setting == group.setting
    data.space_group_t[1, 0] += .125
    with pytest.raises(ValueError, match='pyg_mapping crystal.space_group'):
        Crystal.from_pyg(data)


def test_standalone_empty_and_mixed_loader_save_load():
    cell_only = Crystal(5, 6, 7, 90, 90, 90, space_group=SpaceGroup(1))
    empty = Crystal(5, 6, 7, 90, 90, 90, molecule=Molecule(), space_group=SpaceGroup(1))
    complete = crystal(images=(ZERO, ZERO))
    molecular = molecule()
    objects = (cell_only, empty, complete, molecular)
    graphs = [item.to_pyg() for item in objects]
    batches = list(DataLoader(graphs, batch_size=4, shuffle=False))
    assert len(batches) == 1 and batches[0].num_graphs == 4
    buffer = io.BytesIO();torch.save(batches[0], buffer);buffer.seek(0)
    batch = torch.load(buffer, map_location='cpu', weights_only=False)
    split = batch.to_data_list()
    first, second, third = (Crystal.from_pyg(item) for item in split[:3])
    assert first.molecule is None
    assert second.molecule is not None and len(second.molecule.atoms) == 0
    assert _read_crystal_record(third) == _read_crystal_record(complete)
    assert Molecule.from_pyg(split[3]).crystal is None
    with pytest.raises(ValueError, match='explicit oriented cell'):
        Crystal.from_pyg(split[3])


def test_foreign_explicit_matrix_and_images_do_not_infer_a_group():
    data = Data(num_nodes=2, z=torch.tensor([6, 8]), edge_index=torch.tensor([[0], [1]]),
                pos=torch.tensor([[0., 0., 0.], [1., 0., 0.]]),
                basis=torch.eye(3, dtype=torch.float64).reshape(1, 3, 3),
                voltage=torch.tensor([[1, -2, 0]]), filled=torch.tensor([0., .5], dtype=torch.float64))
    mapping = {'fields': {'z': 'z', 'pos': 'pos', 'edge_index': 'edge_index', 'cell_matrix': 'basis',
                          'bond_image': 'voltage', 'occupancy': 'filled'},
               'edge_mode': 'single', 'coordinate_unit': 'nanometer'}
    value = Crystal.from_pyg(data, field_map=mapping)
    np.testing.assert_array_equal(value.matrix, np.eye(3) * 10)
    assert value.space_group is None and value.bond_images == ((1, -2, 0),)
    assert value.site_relations['occupancies'] == (0., .5)
    missing = deepcopy(mapping);del missing['fields']['bond_image']
    with pytest.warns(RuntimeWarning, match='pyg_unmapped_field voltage'):
        unknown = Crystal.from_pyg(data, field_map=missing)
    assert unknown.bond_images == (None,)


def test_foreign_paired_images_are_opposites_and_preserve_unknown():
    data = Data(num_nodes=2, z=torch.tensor([6, 8]), edge_index=torch.tensor([[0, 1], [1, 0]]),
                basis=torch.eye(3, dtype=torch.float64).reshape(1, 3, 3),
                voltage=torch.tensor([[1, -2, 0], [-1, 2, 0]]))
    mapping = {'fields': {'z': 'z', 'edge_index': 'edge_index', 'cell_matrix': 'basis', 'bond_image': 'voltage'},
               'edge_mode': 'paired', 'coordinate_unit': 'angstrom'}
    result = Crystal.from_pyg(data, field_map=mapping)
    assert result.bond_images == ((1, -2, 0),)
    data.voltage[1, 0] = 3
    with pytest.raises(ValueError, match='paired bond_image'):
        Crystal.from_pyg(data, field_map=mapping)


def test_foreign_exact_context_and_contradictory_selected_values():
    value = crystal(images=(ZERO, ZERO))
    data = value.to_pyg()
    foreign = Data(num_nodes=3, z=data.z, edge_index=data.edge_index, exact=_crystal_to_data(_read_crystal_record(value)))
    mapping = {'fields': {'z': 'z', 'edge_index': 'edge_index', 'crystal_context': 'exact'},
               'edge_mode': 'single', 'coordinate_unit': 'angstrom'}
    restored = Crystal.from_pyg(foreign, field_map=mapping)
    assert _read_crystal_record(restored) == _read_crystal_record(value)
    foreign.basis = data.cell_matrix * 2
    mapping['fields']['cell_matrix'] = 'basis'
    with pytest.raises(ValueError, match='contradict exact context'):
        Crystal.from_pyg(foreign, field_map=mapping)


@pytest.mark.parametrize('field', ('cell_matrix', 'bond_image', 'bond_image_present',
                                   'occupancy', 'occupancy_present', 'space_group_R', 'space_group_t', 'hotpot_crystal'))
def test_missing_required_current_fields_reject_locally(field):
    data = crystal().to_pyg()
    del data[field]
    with pytest.raises((ValueError, KeyError), match='pyg_|hotpot_crystal'):
        Crystal.from_pyg(data)


def test_stale_context_and_numeric_group_mismatch_do_not_mutate_source():
    value = crystal();before = state(value);data = value.to_pyg()
    data.hotpot_crystal[0]['current_atom_keys'][0] = 99
    with pytest.raises(ValueError, match='pyg_mapping crystal'):
        Crystal.from_pyg(data)
    assert state(value) == before
    data = value.to_pyg();data.space_group_R[0, 0, 0] = 17
    with pytest.raises(ValueError, match='pyg_mapping crystal.space_group'):
        Crystal.from_pyg(data)
    assert state(value) == before


def test_restored_context_drives_actual_finite_expansion():
    value = Crystal(8, 9, 10, 90, 90, 90, molecule=molecule(),
                    space_group=SpaceGroup(1), bond_images=(ZERO, ZERO))
    restored = Crystal.from_pyg(value.to_pyg())
    generated = restored.expand((2, 1, 1))
    units = generated.unit_molecules()
    assert len(units) == 2 and all(len(part.atoms) == 3 for part in units)
    assert generated.matrix[0, 0] == pytest.approx(16.)
    assert len(value.molecule.atoms) == 3


def test_nonzero_voltage_remains_periodic_and_independent_finite_content_survives():
    mol = molecule()
    mol.add_bond(2, 0, bond_order=1)
    mol.create_atom(atomic_number=8, coordinates=(3., 3., 3.), formal_charge=0,
                    implicit_hydrogens=0)
    value = Crystal(8, 9, 10, 90, 90, 90, molecule=mol,
                    space_group=SpaceGroup(1), bond_images=((1, 0, 0), ZERO, ZERO))
    restored = Crystal.from_pyg(value.to_pyg())
    with pytest.warns(RuntimeWarning, match='CRYSTAL-MOLECULAR-LIFT-UNAVAILABLE'):
        finite = restored.unit_molecules()
    assert len(finite) == 1 and len(finite[0].atoms) == 1
    assert finite[0].atoms[0].atomic_number == 8
    assert len(restored.molecule.bonds) == 3
    assert restored.bond_images == ((1, 0, 0), ZERO, ZERO)


def test_edited_positions_feed_actual_crystal_geometry_without_repair():
    mol = Molecule()
    for number, point in ((6, (0., 0., 0.)), (8, (2., 0., 0.))):
        mol.create_atom(atomic_number=number, coordinates=point, formal_charge=0,
                        implicit_hydrogens=0)
    value = Crystal(8, 9, 10, 90, 90, 90, molecule=mol, space_group=SpaceGroup(1))
    data = value.to_pyg()
    data.pos[1, 0] = .125
    edited = Crystal.from_pyg(data)
    with pytest.warns(RuntimeWarning, match='CRYSTAL-CROWDING'):
        findings = edited.geometry_findings(coincidence_tolerance=1e-8, close_contact_cutoff=.5)
    assert any(row[0] == 'CRYSTAL-CROWDING' for row in findings)
    assert edited.molecule.coordinates[1, 0] == .125
    assert value.molecule.coordinates[1, 0] == 2.
