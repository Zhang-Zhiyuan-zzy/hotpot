"""Independent molecular facts, edits and actual ordinary PyG workflows."""

from dataclasses import replace
import io
import numpy as np
import pytest

from hotpot import BondKind, Crystal, Molecule
from hotpot.cheminfo.structure._fields import ATOM_SOURCE_CODES, FIELD_INDEX, FIELD_PRESENCE_BITS
from hotpot.cheminfo.structure._identity import _capture_source
from hotpot.cheminfo.structure._results import (
    _capture_mca_source, _import_partial_charges, _select_mca_result,
)
from hotpot.cheminfo.structure.stereo import (
    AtomTransportRecord, BondTransportRecord, HydrogenTransportState, StereoAnchor,
    StereoAnchorKind, StereoCarrier, StereoCarrierKind, StereoGroupKind, StereoGroupRef,
    StereoPayload, StereoRecord,
)
from hotpot.cheminfo.operations.chemistry import _set_stereo_payload
from hotpot.cheminfo.operations.inference.mca.results import AtomPrediction, MoleculePrediction, SitePrediction
from hotpot.cheminfo.structure.conformer import (
    ConformerBatch, Conformers, _import_frame_values, _publish_conformer_ensemble,
)

torch = pytest.importorskip('torch')
from torch_geometric.data import Data, Batch
from torch_geometric.loader import DataLoader


def molecule():
    mol = Molecule()
    old = mol.create_atom(atomic_number=1)
    mol.remove_atom(old)
    for i, z in enumerate((6, 8, 7, 30)):
        mol.create_atom(atomic_number=z, formal_charge=0, implicit_hydrogens=(3, 0, 1, 0)[i],
                        coordinates=(i * 1.25, i % 2, 0.), id=91 + i)
    mol.add_bond(0, 1, 1.)
    mol.add_bond(1, 2, 2.25, bond_kind=BondKind.UNKNOWN, bond_source='fixture',
                 bond_source_metadata={'values': np.array([0., 2.25]), 'tag': ('fractional', 3)})
    mol.add_bond(2, 3, 1., bond_kind=BondKind.DATIVE, bond_direction='atom1_to_atom2',
                 bond_source='fixture', bond_source_metadata={'kind': 'coordination'})
    mol.atoms[0].x_constraint = True
    mol.bonds[1].constraint = True
    mol.bonds[1].id = 731
    mol.angles[0].constraint = True
    mol.torsions[0].constraint = True
    mol._pair_wiberg_bond_orders[(mol.atoms[0]._uid, mol.atoms[2]._uid)] = 0.125
    mol.atoms[1].__dict__['frac_valence'] = 1.75
    mol.charge = 0
    mol.spin_mult = 2
    mol.identifier = 'independent'
    mol.properties = {'target': 0., 'array': np.array([1., 2.]), 'tuple': (b'raw', None)}
    mol._crystal = Crystal(5., 6., 7., 90., 100., 110.)
    return mol


def stereo(mol):
    current = HydrogenTransportState(1, 2, True)
    source = HydrogenTransportState(8, 9, False)
    fact = AtomTransportRecord(0, 2**100, -(2**99), current, source, 'fixture')
    payload = StereoPayload(1, (StereoRecord(StereoAnchor(StereoAnchorKind.ATOM, (0,)),
        (StereoCarrier(StereoCarrierKind.ATOM, 1), StereoCarrier(StereoCarrierKind.ATOM, 2),
         StereoCarrier(StereoCarrierKind.IMPLICIT_H)), 'opaque',
        StereoGroupRef('fixture-scope', StereoGroupKind.AND, -(2**100)), 'fixture', 'opaque'),),
        (fact, fact), (BondTransportRecord((0, 1), 'opaque-display'),))
    _set_stereo_payload(mol, payload)
    return payload


def raw(mol):
    return mol._capture_factual_record(include_frames=False)[0]


def test_independent_complete_facts_and_single_bond_authority():
    mol = molecule()
    data = mol.to_pyg()
    assert type(data) is Data and not hasattr(Molecule, 'to_pyg_data')
    assert data.z.tolist() == [6, 8, 7, 30]
    assert data.atom_key.tolist() == [0, 1, 2, 3] != [a._uid for a in mol.atoms]
    assert data.edge_index.T.tolist() == [[0, 1], [1, 2], [2, 3]]
    assert data.bond_kind.tolist() == [2, 0, 6]
    assert data.bond_order.tolist() == [1., 2.25, 1.]
    assert data.bond_direction.tolist() == [0, 0, 1]
    assert data.wiberg_index.tolist() == [[0], [2]] and data.wiberg_value.tolist() == [.125]
    assert data.angle_index.shape[0] == 3 and data.torsion_index.shape == (4, 1)
    restored = Molecule.from_pyg(data)
    assert [a.atomic_number for a in restored.atoms] == [6, 8, 7, 30]
    assert [b.bond_order for b in restored.bonds] == [1., 2.25, 1.]
    assert restored.bonds[1].bond_kind is BondKind.UNKNOWN and restored.bonds[1].id == 731
    assert restored.bonds[1].constraint and restored.atoms[0].x_constraint
    assert restored.angles[0].constraint and restored.torsions[0].constraint
    assert restored.bonds[2].bond_direction == 'atom1_to_atom2'
    assert restored._charge_present and restored.charge == 0 and restored.spin_mult == 2
    assert restored.identifier == 'independent'
    assert restored._owner_token is not mol._owner_token
    assert restored.properties['tuple'] == (b'raw', None)
    np.testing.assert_equal(restored.properties['array'], [1., 2.])
    np.testing.assert_equal(restored.bonds[1].bond_source_metadata['values'], [0., 2.25])
    assert restored.atoms[1].frac_valence == 1.75
    assert restored._crystal.beta == 100.


def test_missing_facts_remain_distinct_from_explicit_zero_and_isolated_atoms():
    mol = Molecule()
    mol.create_atom(atomic_number=6)
    mol.create_atom(atomic_number=8, formal_charge=0, coordinates=(0., 0., 0.))
    data = mol.to_pyg()
    assert data.num_nodes == 2 and data.edge_index.shape == (2, 0)
    assert data.atom_present[:, 1].tolist() == [False, True]
    assert data.atom_present[:, 5:].tolist() == [[False] * 3, [True] * 3]
    other = Molecule.from_pyg(data)
    mask = np.frombuffer(raw(other).atom_presence, dtype='<u4')
    assert not mask[0] & FIELD_PRESENCE_BITS[FIELD_INDEX['formal_charge']]
    assert mask[1] & FIELD_PRESENCE_BITS[FIELD_INDEX['formal_charge']]
    assert not other._charge_present and not other._spin_present


def test_stereo_occurrences_large_integers_and_distinct_hydrogen_authorities():
    mol = molecule(); expected = stereo(mol)
    data = mol.to_pyg()
    assert data.total_h[0] == 3
    assert data.hotpot_stereo[0]['atom_facts'][0]['current_hydrogens']['implicit_hydrogens'] == 1
    other = Molecule.from_pyg(data)
    assert raw(other).stereo_payload == expected
    assert other.atoms[0].isotope == 2**100
    data.total_h[0] = 17
    changed = Molecule.from_pyg(data)
    payload = raw(changed).stereo_payload
    assert payload.atom_facts[0].current_hydrogens == HydrogenTransportState(1, 2, True)
    assert payload.atom_facts[0].source_hydrogens == HydrogenTransportState(8, 9, False)
    assert np.frombuffer(raw(changed).atom_values, dtype='<f8').reshape(4, 19)[0, 14] == 17


def test_edited_tensors_and_reordered_members_drive_reconstruction():
    mol = molecule(); stereo(mol)
    data = mol.to_pyg()
    order = torch.tensor([2, 0, 3, 1]); inverse = torch.argsort(order)
    for name in ('atom_key', 'z', 'formal_charge', 'total_h', 'aromatic', 'atom_id', 'pos',
                 'atom_present', 'atom_source', 'representation_available', 'atom_constraint'):
        data[name] = data[name][order]
    for name in ('edge_index', 'angle_index', 'torsion_index', 'wiberg_index'):
        data[name] = inverse[data[name]]
    edge_order = torch.tensor([2, 0, 1])
    data.edge_index = data.edge_index[:, edge_order]
    for name in ('bond_key', 'bond_order', 'bond_kind', 'bond_direction', 'bond_aromatic', 'bond_id', 'bond_constraint'):
        data[name] = data[name][edge_order]
    data.pos[1] = torch.tensor([8., 9., 10.])
    data.formal_charge[0] = -1
    data.net_charge[0] = .25
    other = Molecule.from_pyg(data)
    assert [a.atomic_number for a in other.atoms] == [7, 6, 30, 8]
    np.testing.assert_equal(other.coordinates[1], [8., 9., 10.])
    assert other.atoms[0].formal_charge == -1 and other.charge == .25
    assert other.atoms[1].isotope == 2**100
    assert other.bonds[0].bond_kind is BondKind.DATIVE
    assert other.bonds[2].id == 731
    assert raw(other).stereo_payload.records[0].anchor.atom_indices == (1,)


def test_export_and_import_do_not_prepare_chemistry(monkeypatch):
    from hotpot.cheminfo.operations import chemistry
    mol = molecule()
    def forbidden(*args, **kwargs):
        pytest.fail('transport prepared chemistry')
    for name in ('_prepare_hydrogens', '_prepare_aromatic_representation', '_calculate_aromatic_assignment'):
        monkeypatch.setattr(chemistry, name, forbidden)
    before = raw(mol)
    data = mol.to_pyg()
    assert not data.representation_available.any()
    other = Molecule.from_pyg(data)
    assert raw(mol).atom_values == before.atom_values
    assert len(other.atoms) == 4


def test_no_source_or_destination_buffer_aliases():
    mol = molecule(); data = mol.to_pyg(); other = Molecule.from_pyg(data)
    before = mol.coordinates.copy(); restored = other.coordinates.copy()
    data.pos[0, 0] = 999
    data.hotpot_graph[0]['annotations']['items'][1][1]['tensor'][0] = 44
    np.testing.assert_equal(mol.coordinates, before)
    np.testing.assert_equal(other.coordinates, restored)
    np.testing.assert_equal(mol.properties['array'], [1., 2.])
    np.testing.assert_equal(other.properties['array'], [1., 2.])
    other.atoms[0].x = -100
    assert mol.coordinates[0, 0] == 0


def _mca(mol):
    atoms = tuple(AtomPrediction(i, atom.symbol, 10. + i) for i, atom in enumerate(mol.atoms))
    result = MoleculePrediction('fixture', 0, (SitePrediction(1, 'O', 'fixture', 11.),), 'fixture-v1', atoms)
    _select_mca_result(mol, result, _capture_mca_source(mol), tuple(atom._uid for atom in mol.atoms))


def test_geometry_stale_mca_has_proven_join_and_representation_stale_does_not():
    mol = molecule(); _mca(mol)
    mol.atoms[0].x = 7
    observation = mol.to_pyg().hotpot_observations[0]['current_source'][0]
    assert observation['subjects']['atom_keys'] == [0, 1, 2, 3]
    assert observation['invalidated_by'] == ['geometry']
    mol.atoms[0].formal_charge = 1
    data = mol.to_pyg()
    observation = data.hotpot_observations[0]['current_source'][0]
    assert observation['subjects']['atom_keys'] == [None] * 4
    other = Molecule.from_pyg(data)
    assert other._mca_result is None
    retained = other.properties['pyg_observations']['observations']['current_source'][0]
    assert retained['prediction']['sites'][0]['atom_index'] == 1
    assert retained['prediction']['model_variant'] == 'fixture-v1'
    again = other.to_pyg().hotpot_observations[0]
    assert again['current_source'] == [] and again['historical'] is not None


def test_partial_charge_removed_subjects_are_history_without_private_uids():
    mol = molecule()
    _import_partial_charges(mol, [0., None, -.25, .25], producer=ATOM_SOURCE_CODES['explicit'], method='supplied')
    mol.remove_atom(0)
    data = mol.to_pyg()
    observation = data.hotpot_observations[0]['current_source'][0]
    assert observation['subjects']['ordinals'] == [0, 1, 2, 3]
    assert observation['subjects']['atom_keys'] == [None, 0, 1, 2]
    assert 'atom_uids' not in observation['subjects']
    assert observation['values'] == [0., None, -.25, .25]
    other = Molecule.from_pyg(data)
    assert other._partial_charge_result is None
    record = raw(other)
    masks = np.frombuffer(record.atom_presence, dtype='<u4')
    assert not (masks & FIELD_PRESENCE_BITS[FIELD_INDEX['partial_charge']]).any()


def test_source_and_generated_frames_retain_measured_values_without_selection():
    mol = molecule()
    coordinates = np.stack((mol.coordinates, mol.coordinates + 1))
    values = (coordinates, [0., 2.], [[0., .1, .2, .3], [1., 1.1, 1.2, 1.3]], None,
              np.arange(24.).reshape(2, 4, 3), None, None, None, None)
    _import_frame_values(mol, values, 0, producer='fixture', method='measured', units=(('energy', 'eV'),))
    _publish_conformer_ensemble(mol, ConformerBatch(coordinates + 3, (4, 9)), {'energy': [7., 8.]}, 1,
        input_source=_capture_source(mol, ('geometry',)), method='fixture-optimizer', units=(('energy', 'kJ/mol'),))
    data = mol.to_pyg()
    assert data.hotpot_frames[0]['generated'][0]['ordinals'] == [4, 9]
    assert 'input_source' not in data.hotpot_frames[0]['generated'][0]
    data.pos += 20
    other = Molecule.from_pyg(data)
    assert len(other._source_frames) == 2 and other._conformer_results == () and other._frame_result is None
    np.testing.assert_equal(other.coordinates, data.pos.numpy())
    assert other.conformers.energy(0) == 0.
    np.testing.assert_equal(other.conformers.partial_charge(1), [1., 1.1, 1.2, 1.3])
    assert other.energy is None and other._partial_charge_result is None
    assert other.properties['pyg_observations']['frames']['generated'][0]['ordinals'] == [4, 9]


def test_missing_stereo_or_metadata_members_fail_locally():
    data = molecule().to_pyg(); data.bond_key[0] = 999
    with pytest.raises(ValueError, match='pyg_mapping hotpot_bonds'):
        Molecule.from_pyg(data)
    mol = molecule(); stereo(mol); data = mol.to_pyg()
    data.edge_index[:, 0] = torch.tensor([0, 2])
    with pytest.raises(ValueError, match='pyg_stereo_reference'):
        Molecule.from_pyg(data)
    data = mol.to_pyg(); data.atom_key[0] = 100
    with pytest.raises(ValueError, match='pyg_stereo_reference'):
        Molecule.from_pyg(data)


def test_unsupported_actual_metadata_has_exact_local_path():
    mol = molecule(); mol.properties['unsupported'] = object()
    with pytest.raises(TypeError, match=r"pyg_unsupported_metadata properties\['unsupported'\]"):
        mol.to_pyg()
    mol = molecule(); mol.bonds[0]._bond_source_metadata['unsupported'] = object()
    with pytest.raises(TypeError, match=r"pyg_unsupported_metadata bonds\[0\].metadata\['unsupported'\]"):
        mol.to_pyg()


def test_foreign_map_explicit_columns_units_absence_and_precise_unmapped_warning():
    data = Data(num_nodes=2, x=torch.tensor([[6., 99.], [8., 98.]]),
        pos=torch.tensor([[0., 0., 0.], [1., 0., 0.]]),
        edge_index=torch.tensor([[0], [1]]), edge_attr=torch.tensor([[2.25]]), target=torch.tensor([4.]))
    mapping = {'fields': {'z': ('x', 0), 'pos': 'pos', 'edge_index': 'edge_index',
        'bond_order': ('edge_attr', 0), 'annotations': 'target'}, 'edge_mode': 'single', 'coordinate_unit': 'nanometer'}
    with pytest.warns(RuntimeWarning, match=r'pyg_unmapped_field x\[:,1\]'):
        other = Molecule.from_pyg(data, field_map=mapping)
    assert [a.atomic_number for a in other.atoms] == [6, 8]
    assert other.coordinates[1, 0] == 10.
    assert other.bonds[0].bond_kind is BondKind.UNKNOWN and other.bonds[0].bond_order == 2.25
    assert not other._charge_present
    np.testing.assert_equal(other.properties['annotations'], np.array([4.]))


def test_foreign_paired_arcs_require_exact_agreement_and_preserve_first_order():
    data = Data(z=torch.tensor([6, 8, 7]), edge_index=torch.tensor([[1, 2, 2, 1], [2, 1, 0, 2]]))
    data.edge_index = torch.tensor([[1, 2, 2, 0], [2, 1, 0, 2]])
    data.order = torch.tensor([2., 2., 1., 1.], dtype=torch.float64)
    data.direction = torch.tensor([1, -1, 0, 0], dtype=torch.int8)
    mapping = {'fields': {'z': 'z', 'edge_index': 'edge_index', 'bond_order': 'order', 'bond_direction': 'direction'},
               'edge_mode': 'paired', 'coordinate_unit': None}
    other = Molecule.from_pyg(data, field_map=mapping)
    assert [(b.a1idx, b.a2idx) for b in other.bonds] == [(1, 2), (2, 0)]
    assert other.bonds[0].bond_direction == 'atom1_to_atom2'
    data.order[1] = 3.
    with pytest.raises(ValueError, match='pyg_mapping paired bond_order'):
        Molecule.from_pyg(data, field_map=mapping)


def test_schema_errors_do_not_publish_or_mutate_input():
    data = molecule().to_pyg(); before = data.pos.clone()
    data.atom_key[1] = data.atom_key[0]
    with pytest.raises(ValueError, match='pyg_mapping atom_key'):
        Molecule.from_pyg(data)
    assert torch.equal(data.pos, before)
    with pytest.raises(ValueError, match='pyg_schema'):
        Molecule.from_pyg(Data(z=torch.tensor([6])))
    batch = Batch.from_data_list([Molecule().to_pyg(), molecule().to_pyg()])
    with pytest.raises(ValueError, match='pyg_mapping Batch'):
        Molecule.from_pyg(batch)


def test_real_batch_loader_cpu_serialization_and_small_training_path():
    mol = molecule(); stereo(mol)
    single = Molecule(); single.create_atom(atomic_number=1)
    graphs = [mol.to_pyg(), Molecule().to_pyg(), single.to_pyg()]
    batch = next(iter(DataLoader(graphs, batch_size=3, shuffle=False))).cpu()
    assert batch.ptr.tolist() == [0, 4, 4, 5]
    parts = batch.to_data_list()
    assert [part.num_nodes for part in parts] == [4, 0, 1]
    assert parts[0].hotpot_stereo[0]['atom_facts'][0]['isotope'] == 2**100
    restored = [Molecule.from_pyg(part) for part in parts]
    assert [len(mol.atoms) for mol in restored] == [4, 0, 1]
    assert restored[0].properties['tuple'] == (b'raw', None)
    stream = io.BytesIO();torch.save(batch, stream);stream.seek(0)
    loaded = torch.load(stream, weights_only=False)
    assert loaded.to_data_list()[1].num_nodes == 0
    model = torch.nn.Embedding(31, 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=.01)
    initial = model.weight.detach().clone()
    output = torch.zeros(3, 1).index_add(0, batch.batch, model(batch.z))
    loss = torch.nn.functional.mse_loss(output.flatten(), torch.tensor([2., 0., 1.]))
    loss.backward();optimizer.step()
    assert not torch.equal(initial, model.weight)
    assert parts[0].z.storage_offset() + parts[0].z.numel() <= parts[2].z.storage_offset()


@pytest.mark.parametrize('wrapper,field', [
    ('hotpot_graph', 'annotations'), ('hotpot_frames', 'generated'),
    ('hotpot_observations', 'current_source'),
])
def test_missing_structured_schema_fields_have_named_local_errors(wrapper, field):
    data = molecule().to_pyg()
    del data[wrapper][0][field]
    with pytest.raises(ValueError, match=rf'pyg_schema {wrapper}\.{field}'):
        Molecule.from_pyg(data)


def test_partial_charge_subject_alignment_is_validated():
    mol = molecule()
    _import_partial_charges(mol, [0., None, -.25, .25], producer=ATOM_SOURCE_CODES['explicit'])
    data = mol.to_pyg()
    data.hotpot_observations[0]['current_source'][0]['values'].pop()
    with pytest.raises(ValueError, match='pyg_mapping observations.partial_charge'):
        Molecule.from_pyg(data)


def test_generated_frame_shape_is_validated_even_when_historical():
    mol = molecule()
    _publish_conformer_ensemble(mol, ConformerBatch(mol.coordinates[None], (7,)), {}, 0,
        input_source=_capture_source(mol, ('geometry',)), method='fixture-optimizer')
    data = mol.to_pyg()
    data.hotpot_frames[0]['generated'][0]['coordinates'] = torch.zeros(1, 3, 3)
    with pytest.raises(ValueError, match='pyg_schema generated_frames.coordinates'):
        Molecule.from_pyg(data)


def test_portable_numpy_values_preserve_dtype_shape_and_independence():
    mol = Molecule()
    mol.properties = {'scalar': np.int32(7), 'bool': np.array([True, False]),
        'unsigned': np.array([2**63 + 4], dtype=np.uint64),
        'complex': np.array([[1 + 2j]], dtype=np.complex64),
        'extended': np.array([1.25], dtype=np.longdouble),
        'endian': np.array([4.], dtype='>f8')}
    data = mol.to_pyg()
    other = Molecule.from_pyg(Batch.from_data_list([data]).cpu().to_data_list()[0])
    for key, value in mol.properties.items():
        assert other.properties[key].dtype == value.dtype
        np.testing.assert_equal(other.properties[key], value)
    other.properties['unsigned'][0] = 0
    assert mol.properties['unsigned'][0] == 2**63 + 4


@pytest.mark.parametrize('selector', [17, ('x', -1), ('x', 2)])
def test_foreign_invalid_selector_has_named_local_error(selector):
    data = Data(x=torch.tensor([[6., 0.]]))
    mapping = {'fields': {'z': selector}, 'edge_mode': 'single', 'coordinate_unit': None}
    with pytest.raises(ValueError, match='pyg_(schema|mapping).*z'):
        Molecule.from_pyg(data, field_map=mapping)


def test_foreign_paired_mapping_rejects_lossy_discrete_values():
    data = Data(z=torch.tensor([6, 8]), edge_index=torch.tensor([[0, 1], [1, 0]]),
        constraint=torch.tensor([1., 2.]))
    mapping = {'fields': {'z': 'z', 'edge_index': 'edge_index', 'bond_constraint': 'constraint'},
        'edge_mode': 'paired', 'coordinate_unit': None}
    with pytest.raises(ValueError, match='pyg_mapping bond_constraint'):
        Molecule.from_pyg(data, field_map=mapping)



def test_exact_schema_requires_explicit_node_count_and_accepts_one_graph_batch():
    data = molecule().to_pyg()
    assert len(Molecule.from_pyg(Batch.from_data_list([data])).atoms) == 4
    del data.num_nodes
    with pytest.raises(ValueError, match='pyg_schema num_nodes'):
        Molecule.from_pyg(data)


def test_removed_source_frame_subjects_remain_historical_without_guessing_a_join():
    mol = molecule()
    original = mol.coordinates.copy()
    values = (original[None], [0.], [[0., .1, .2, .3]], None, None, None, None, None, None)
    _import_frame_values(mol, values, 0, producer='fixture', method='measured')
    mol.remove_atom(0)
    data = mol.to_pyg()
    frame = data.hotpot_frames[0]['sources'][0]
    assert frame['subjects']['atom_keys'] == [None, 0, 1, 2]
    other = Molecule.from_pyg(data)
    assert other._frame_result is None and other._partial_charge_result is None
    assert other.conformers.energy(0) is None
    np.testing.assert_equal(other.conformers.partial_charge(0), [.1, .2, .3])
    np.testing.assert_equal(other.properties['pyg_observations']['frames']['sources'][0]['coordinates'], original)
    assert other.properties['pyg_observations']['frames']['sources'][0]['subjects']['atom_keys'] == [None, 0, 1, 2]



@pytest.mark.parametrize('field,values,dtype', [
    ('z', [6.5, 8.], torch.float64),
    ('formal_charge', [.5, 0.], torch.float64),
    ('atom_key', [float(2**63), 0.], torch.float64),
    ('atom_key', [2**63 + 1, 0], torch.uint64),
    ('atom_source', [[256] * 8, [1] * 8], torch.int64),
    ('aromatic', [0., 2.], torch.float64),
    ('atom_present', [[1] * 8, [2] * 8], torch.int64),
    ('atom_constraint', [[0, 0, 0], [0, 2, 0]], torch.int64),
    ('bond_direction', [128], torch.int64),
    ('bond_kind', [2.5], torch.float64),
    ('bond_constraint', [2], torch.int64),
    ('net_charge_present', [2.], torch.float64),
])
def test_foreign_discrete_mapping_rejects_value_loss_and_numeric_overflow(field, values, dtype):
    data = Data(num_nodes=2, z=torch.tensor([6, 8]), edge_index=torch.tensor([[0], [1]]))
    data.mapped = torch.tensor(values, dtype=dtype)
    mapping = {'fields': {'z': 'z', 'edge_index': 'edge_index', field: 'mapped'},
               'edge_mode': 'single', 'coordinate_unit': None}
    with pytest.raises(ValueError, match=rf'pyg_mapping {field}: conversion'):
        Molecule.from_pyg(data, field_map=mapping)


def test_foreign_exact_integral_floats_boolean_zero_one_and_continuous_values_remain_supported():
    data = Data(num_nodes=2, z=torch.tensor([6., 8.], dtype=torch.float64),
        formal_charge=torch.tensor([0., -1.], dtype=torch.float64),
        aromatic=torch.tensor([0., 1.], dtype=torch.float64),
        atom_key=torch.tensor([2**63 - 1, 0]),
        atom_constraint=torch.tensor([[0., 1., 0.], [1., 0., 1.]], dtype=torch.float64),
        edge_index=torch.tensor([[0], [1]]), bond_order=torch.tensor([2.25], dtype=torch.float32),
        bond_direction=torch.tensor([-1.], dtype=torch.float64),
        bond_constraint=torch.tensor([1.], dtype=torch.float64),
        net_charge=torch.tensor([.25], dtype=torch.float32))
    fields = {name: name for name in data.keys() if name != 'num_nodes'}
    mapping = {'fields': fields, 'edge_mode': 'single', 'coordinate_unit': None}
    other = Molecule.from_pyg(data, field_map=mapping)
    assert [atom.atomic_number for atom in other.atoms] == [6, 8]
    assert other.atoms[1].formal_charge == -1
    assert other.bonds[0].bond_order == 2.25 and other.charge == .25
    assert other.bonds[0].bond_direction == 'atom2_to_atom1' and other.bonds[0].constraint
    exported = other.to_pyg()
    assert exported.aromatic.tolist() == [False, True]
    assert exported.atom_constraint.tolist() == [[False, True, False], [True, False, True]]
    assert other.properties['pyg_observations']['initial_atom_keys'] == [2**63 - 1, 0]
