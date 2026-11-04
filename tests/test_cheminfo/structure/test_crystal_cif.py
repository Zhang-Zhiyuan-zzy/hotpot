"""Independent standard CIF projections and complete current-fact roundtrips."""

import builtins
from copy import deepcopy
from dataclasses import fields, replace
from fractions import Fraction as Q
import json
import sys
from types import ModuleType, SimpleNamespace
import warnings

import numpy as np
import pytest

from hotpot import Crystal, Molecule, SpaceGroup, SymmUnit
from hotpot.cheminfo.adapters.cif import _read_cif, _write_cif
from hotpot.cheminfo.operations import crystal_io
from hotpot.cheminfo.structure._builder import _freeze_value
from hotpot.cheminfo.structure._crystal import _read_crystal_record
from hotpot.cheminfo.structure._fields import ATOM_FIELDS, FIELD_INDEX, FIELD_PRESENCE_BITS
from hotpot.cheminfo.structure._results import PartialChargeSet
from hotpot.cheminfo.structure._stereo_codec import _stereo_from_data, _stereo_to_data
from hotpot.cheminfo.structure.conformer import CoordinateFrame
from hotpot.cheminfo.structure.stereo import (
    AtomTransportRecord, BondTransportRecord, HydrogenTransportState,
    StereoAnchor, StereoAnchorKind, StereoCarrier, StereoCarrierKind,
    StereoGroupRef, StereoGroupKind, StereoPayload, StereoRecord,
)


CELL = '''_cell_length_a 5
_cell_length_b 6
_cell_length_c 7
_cell_angle_alpha 90
_cell_angle_beta 90
_cell_angle_gamma 90
'''
SITES = '''loop_
_atom_site_label
_atom_site_type_symbol
_atom_site_fract_x
_atom_site_fract_y
_atom_site_fract_z
C1 C 0 0 0
N1 N .2 .25 .5
'''
OPS = '''loop_
_space_group_symop_id
_space_group_symop_operation_xyz
1 'x,y,z'
'''


def bare(*, sites=SITES, ops=OPS, extra='', name='sample'):
    return 'data_' + name + '\n' + CELL + ops + sites + extra


def bonds(rows, *, symmetry=True, kind=True):
    tags = ['_geom_bond_atom_site_label_1', '_geom_bond_atom_site_label_2']
    if symmetry:
        tags += ['_geom_bond_site_symmetry_1', '_geom_bond_site_symmetry_2']
    if kind:
        tags += ['_geom_bond_type']
    return 'loop_\n' + '\n'.join(tags) + '\n' + '\n'.join(rows) + '\n'


def rewrite(text, changes):
    name, columns = _read_cif(text)
    columns.update(changes)
    pairs, loops = [], []
    for tag, values in columns.items():
        if len(values) == 1:
            pairs.append((tag, values[0]))
        elif values:
            loops.append(([tag], [[value] for value in values]))
    return _write_cif(name, pairs, loops)


def captured(crystal):
    return crystal_io._capture(crystal)[0]


def assert_current_equal(first, second):
    for field in fields(first):
        left, right = getattr(first, field.name), getattr(second, field.name)
        assert left == right, field.name


def source_cell():
    molecule = Molecule()
    for number, point in zip((6, 7, 8, 1), ((.125, -1.5, 2.), (2., .75, .5), (1., 1., 2.), (-.25, 2., 1.))):
        molecule.create_atom(atomic_number=number, coordinates=point)
    for first, second in ((0, 1), (1, 2), (2, 3)):
        molecule.add_bond(first, second, bond_order=1)
    return Crystal.from_matrix(((0., 5., 0.), (-6., 1., 0.), (1., .5, 7.)), molecule=molecule,
                               space_group=SpaceGroup(1), bond_images=((0, 0, 0), (12, -9, 2), None))


def rich_cell():
    cell = source_cell()
    record = captured(cell)
    values = np.frombuffer(record.atom_values, dtype='<f8').reshape(4, len(ATOM_FIELDS)).copy()
    mask = np.frombuffer(record.atom_presence, dtype='<u4').copy()
    sources = np.frombuffer(record.atom_sources, dtype='u1').reshape(values.shape).copy()
    for row, name, value, producer in ((0, 'formal_charge', 0., 1), (1, 'formal_charge', -1., 4),
                                        (0, 'implicit_hydrogens', 0., 3), (0, 'x_constraint', 1., 1),
                                        (2, 'id', 12345., 2)):
        index = FIELD_INDEX[name]
        values[row, index] = value
        mask[row] |= FIELD_PRESENCE_BITS[index]
        sources[row, index] = producer
    h = HydrogenTransportState(0, 1, True)
    payload = StereoPayload(1, (
        StereoRecord(StereoAnchor(StereoAnchorKind.ATOM, (0,)),
            (StereoCarrier(StereoCarrierKind.ATOM, 1), StereoCarrier(StereoCarrierKind.IMPLICIT_H, -17),
             StereoCarrier(StereoCarrierKind.LONE_PAIR), StereoCarrier(StereoCarrierKind.VACANCY, 2)),
            '@opaque', StereoGroupRef('fixture', StereoGroupKind.OR, -(2**130)), 'manual', 'native'),
        StereoRecord(StereoAnchor(StereoAnchorKind.BOND, (1, 2)), (), 'E?', None, 'fixture', 'bond'),
        StereoRecord(StereoAnchor(StereoAnchorKind.AXIS, (0, 1, 2, 3)), (), 'axis?', None, 'fixture', 'axis'),
    ), (AtomTransportRecord(3, 2, 0, HydrogenTransportState(0, 0, False), h, 'source'),
        AtomTransportRecord(3, 2, 1, None, h, 'second occurrence')),
        (BondTransportRecord((1, 2), '/'), BondTransportRecord((2, 1), '\\')))
    partial = PartialChargeSet((0, 1, 2, 3), (0., -.125, None, .125), 'fixture', '7',
                              (('geometry', 'input'),), 'e', (1, 4, 0, 6))
    metadata = {'bytes': b'\x00\xff', 'typed': np.array([[1, 2], [3, 4]], dtype='>i2'),
                'scalar': np.float32(.5), 'nested': [None, ('a', True)]}
    bonds_raw = np.array(((1., 1., 2.7), (1., 0., 1.25), (np.nan, 0., .3)), dtype='<f8')
    record = replace(record, atom_values=values.tobytes(), atom_presence=mask.tobytes(), atom_sources=sources.tobytes(),
        atom_frac_valences=((1, 2.5),), bond_values=bonds_raw.tobytes(),
        bond_kinds=('single', 'dative', 'unknown'), bond_directions=(None, '1->2', None),
        bond_sources=('manual', 'fixture', None), bond_metadata=tuple(_freeze_value(item) for item in (metadata, {}, None)),
        bond_aromatic=(True, False, False), charge=(True, -1, 'manual'), spin_mult=(True, 2, 'manual'),
        stereo_payload=payload, angle_constraints=((0, 1, 2),), torsion_constraints=((0, 1, 2, 3),),
        identifier=(True, 'current facts'), pair_wiberg_bond_orders=((0, 2, .375),),
        partial_charge_result=(partial, ('geometry',)))
    return Molecule._build_molecule(record).crystal


def test_bare_missing_vs_explicit_zero_has_no_inferred_charge_hydrogens_or_bonds():
    sites = '''loop_
_atom_site_label
_atom_site_type_symbol
_atom_site_fract_x
_atom_site_fract_y
_atom_site_fract_z
C1 C 0 0 0
X1 ? ? . ?
'''
    text = bare(sites=sites)
    crystal = Crystal.from_cif(text)
    record = captured(crystal)
    values, mask = crystal_io._arrays(record)
    assert record.atom_count == 2 and record.bond_endpoints == ()
    assert mask[0] == sum(FIELD_PRESENCE_BITS[FIELD_INDEX[key]] for key in ('atomic_number', 'x', 'y', 'z'))
    assert mask[1] == 0
    assert values[0, FIELD_INDEX['x']] == 0
    assert np.isnan(values[1, FIELD_INDEX['x']])
    assert record.charge == (False, None, None) and record.spin_mult == (False, None, None)
    assert all(not (entry & FIELD_PRESENCE_BITS[FIELD_INDEX['implicit_hydrogens']]) for entry in mask)
    assert crystal.molecule.properties['cif_source'] == {'text': text, 'block': 'sample'}


def test_esds_type_precedence_oxidation_and_d_t_are_source_not_inferred_charge():
    sites = '''loop_
_atom_site_label
_atom_site_type_symbol
_atom_site_fract_x
_atom_site_fract_y
_atom_site_fract_z
Cu1 Fe3+ .10(2) 2.5(7)e-1 -0.5(3)
D1 D 0 0 0
T1 T .2 .3 .4
'''
    crystal = Crystal.from_cif(bare(sites=sites))
    record = captured(crystal)
    values, mask = crystal_io._arrays(record)
    assert tuple(values[:, FIELD_INDEX['atomic_number']]) == (26., 1., 1.)
    np.testing.assert_allclose(crystal.molecule.coordinates[0], (.5, 1.5, -3.5), atol=1e-14)
    assert [(fact.atom, fact.isotope) for fact in record.stereo_payload.atom_facts] == [(1, 2), (2, 3)]
    assert not np.any(mask & FIELD_PRESENCE_BITS[FIELD_INDEX['formal_charge']])
    assert '.10(2)' in crystal.molecule.properties['cif_source']['text']


@pytest.mark.parametrize('label,number,isotope', [('C12A', 6, None), ('Cl2_7', 17, None), ('D2', 1, 2), ('T', 1, 3)])
def test_label_component_zero_without_type_column(label, number, isotope):
    sites = f'loop_\n_atom_site_label\n_atom_site_fract_x\n_atom_site_fract_y\n_atom_site_fract_z\n{label} 0 0 0\n'
    record = captured(Crystal.from_cif(bare(sites=sites)))
    assert crystal_io._arrays(record)[0][0, FIELD_INDEX['atomic_number']] == number
    assert tuple(fact.isotope for fact in record.stereo_payload.atom_facts) == (() if isotope is None else (isotope,))


@pytest.mark.parametrize('species', ['C/N', 'C0.5N0.5', 'X', 'Du', 'Q', 'Fe2O3'])
def test_composite_dummy_species_reject_locally_with_exact_source(species):
    text = bare(sites=SITES.replace('C1 C ', 'C1 ' + species + ' '))
    with pytest.raises(ValueError, match='CRYSTAL-UNSUPPORTED-PROJECTION') as caught:
        Crystal.from_cif(text)
    assert caught.value.cif_source == text and caught.value.cif_block == 'sample'


def test_case_mmcif_aliases_paths_and_exact_block_selection(tmp_path):
    first = bare(name='First')
    second = bare(name='Second').replace('_atom_site_label', '_atom_site.label_atom_id').replace(
        '_atom_site_type_symbol', '_ATOM_SITE.TYPE_SYMBOL').replace('_atom_site_fract_', '_atom_site.fract_')
    text = (first + second).replace('\n', '\r\n')
    path = tmp_path / 'input.cif'
    path.write_text(text)
    for source in (text, path, str(path)):
        result = Crystal.from_cif(source, block='Second')
        assert len(result.molecule.atoms) == 2
        assert result.molecule.properties['cif_source'] == {'text': text, 'block': 'Second'}
    for selected in (None, 'second'):
        with pytest.raises(ValueError, match='CIF block') as caught:
            Crystal.from_cif(text, block=selected)
        assert caught.value.cif_source == text and caught.value.cif_block == selected


def test_nonowned_adp_refinement_fields_stay_in_exact_archive():
    text = bare(extra='_refine_ls_R_factor_gt .037\n_atom_site_aniso_U_11 0.1(2)\n')
    result = Crystal.from_cif(text)
    assert result.molecule.properties['cif_source']['text'] == text
    assert 'refine_ls_R_factor_gt' not in result.molecule.__dict__


def test_incomplete_fractional_triplet_remains_absent_with_precise_finding():
    with pytest.warns(RuntimeWarning, match='CIF site C1.*incomplete fractional triplet'):
        result = Crystal.from_cif(bare(sites=SITES.replace('C1 C 0 0 0', 'C1 C .1 ? 0')))
    _, mask = crystal_io._arrays(captured(result))
    assert mask[0] & crystal_io._COORDINATES == 0


@pytest.mark.parametrize('bad', ['nan', 'inf', '-inf'])
def test_nonfinite_input_geometry_rejects(bad):
    with pytest.raises(ValueError, match='finite'):
        Crystal.from_cif(bare(sites=SITES.replace('C1 C 0 0 0', f'C1 C {bad} 0 0')))


def test_cartesian_requires_actual_explicit_transform_and_checks_dual_coordinates():
    sites = SITES.replace('_atom_site_fract_', '_atom_site.Cartn_')
    text = bare(sites=sites)
    with pytest.raises(ValueError, match='explicit fractional transform'):
        Crystal.from_cif(text)
    transform = ''.join(f'_atom_sites.fract_transf_matrix[{i}][{j}] {1 / (4 + i) if i == j else 0}\n'
                        for i in (1, 2, 3) for j in (1, 2, 3))
    transform += ''.join(f'_atom_sites.fract_transf_vector[{i}] 0\n' for i in (1, 2, 3))
    result = Crystal.from_cif(text + transform)
    np.testing.assert_allclose(result.molecule.coordinates, ((0, 0, 0), (.2, .25, .5)), atol=1e-14)
    both = sites + '_atom_site_fract_x 0\n_atom_site_fract_y 0\n_atom_site_fract_z 0\n'
    # Use one actual site so both supplied coordinate forms are independently complete.
    both = both.replace('N1 N .2 .25 .5\n', '').replace('C1 C 0 0 0', 'C1 C 1 0 0')
    with pytest.raises(ValueError, match='fractional and transformed Cartesian'):
        Crystal.from_cif(bare(sites=both, extra=transform))


def test_original_operator_ids_and_both_endpoint_frames_define_relative_integer_image():
    ops = '''loop_
_space_group_symop_id
_space_group_symop_operation_xyz
9 '-x,-y,-z'
2 'x,y,z'
'''
    crystal = Crystal.from_cif(bare(ops=ops, extra=bonds(['C1 N1 9_555 9_655 S'])))
    assert crystal.space_group.number is None
    assert crystal.bond_images == ((-1, 0, 0),)
    crystal = Crystal.from_cif(bare(ops=ops, extra=bonds(['C1 N1 2_555 2_465 D'])))
    assert crystal.bond_images == ((-1, 1, 0),)
    assert captured(crystal).bond_kinds == ('double',)


@pytest.mark.parametrize('codes,image', [('. .', (0, 0, 0)), ('. ?', None), ('? .', None)])
def test_known_zero_and_unknown_images_stay_distinct(codes, image):
    result = Crystal.from_cif(bare(extra=bonds(['C1 N1 ' + codes + ' S'])))
    assert result.bond_images == (image,)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        assert Crystal.from_cif(result.to_cif()).bond_images == (image,)


def test_bond_without_periodic_information_and_without_type_does_not_invent_values():
    crystal = Crystal.from_cif(bare(extra=bonds(['C1 N1'], symmetry=False, kind=False)))
    record = captured(crystal)
    assert crystal.bond_images == (None,) and record.bond_kinds == ('unknown',)
    assert np.isnan(np.frombuffer(record.bond_values, dtype='<f8')[0])


def test_usual_single_symmetry_two_column_means_first_endpoint_is_unoperated():
    extra = 'loop_\n_geom_bond_atom_site_label_1\n_geom_bond_atom_site_label_2\n_geom_bond_site_symmetry_2\n_ccdc_geom_bond_type\nC1 N1 . A\n'
    result = Crystal.from_cif(bare(extra=extra))
    assert result.bond_images == ((0, 0, 0),)
    assert captured(result).bond_kinds == ('aromatic',)


@pytest.mark.parametrize('rows,ops,reason', [
    (['C1 C1 . . S'], OPS, 'self/parallel'),
    (['C1 N1 . . S', 'N1 C1 . . S'], OPS, 'self/parallel'),
    (['C1 N1 . . S', 'C1 N1 . 1_655 S'], OPS, 'self/parallel'),
    (['C1 Missing . . S'], OPS, 'absent site'),
    (['C1 N1 . 77_555 S'], OPS, 'unknown CIF Bond'),
    (['C1 N1 . . X'], OPS, 'unmapped CIF Bond'),
    (['C1 N1 . 2_555 S'], OPS + "2 '-x,-y,-z'\n", 'nontranslational'),
    (['C1 N1 . 2_555 S'], OPS + "2 'x+1/2,y,z'\n", 'nontranslational'),
])
def test_unsupported_bond_projection_rejects_full_input_without_collapsing(rows, ops, reason):
    text = bare(ops=ops, extra=bonds(rows))
    with pytest.raises(ValueError, match=reason) as caught:
        Crystal.from_cif(text)
    assert caught.value.cif_source == text


@pytest.mark.parametrize('kind,order', [('S', 1.), ('D', 2.), ('T', 3.), ('A', 1.5)])
def test_standard_type_mapping_is_explicit(kind, order):
    result = Crystal.from_cif(bare(extra=bonds(['C1 N1 . . ' + kind])))
    assert np.frombuffer(captured(result).bond_values, dtype='<f8')[0] == order


@pytest.mark.parametrize('number', [1, 2, 4, 5, 19, 75, 146])
def test_explicit_reference_operations_bind_only_verified_exact_setting(number):
    group = SpaceGroup(number)
    matrix = (((5., 0., 0.), (-2.5, 5 * np.sqrt(3) / 2, 0.), (0., 0., 7.))
              if number == 146 else np.diag((5., 5., 7.)))
    crystal = Crystal.from_matrix(matrix, space_group=group)
    result = Crystal.from_cif(crystal.to_cif())
    assert result.space_group == group
    assert result.molecule is None


def test_label_only_group_stays_unknown_and_changed_origin_is_not_relabelled_reference():
    with pytest.warns(RuntimeWarning, match='active group unknown'):
        result = Crystal.from_cif(bare(ops='', extra='_space_group_IT_number 2\n'))
    assert result.space_group is None and result.symm_units is None
    ops = OPS + "2 '-x+2/7,-y-4/5,-z+6/11'\n"
    with pytest.warns(RuntimeWarning, match='remain anonymous'):
        result = Crystal.from_cif(bare(ops=ops, extra='_space_group_IT_number 2\n'))
    assert result.space_group.number is None and result.space_group.setting is None
    assert len(result.symm_units) == 2


def test_native_changed_setting_and_centered_primitive_setting_keep_full_binding():
    for group, matrix in (
        (SpaceGroup(2).change_setting(((1, 0, 0), (0, 1, 0), (0, 0, 1)), (Q(1, 7), Q(-2, 5), Q(3, 11))), np.diag((5., 6., 7.))),
        (SpaceGroup(5).change_setting(((Q(1, 2), Q(-1, 2), 0), (Q(1, 2), Q(1, 2), 0), (0, 0, 1)), (0, 0, 0)),
         ((2.5, 2.5, 0.), (-2.5, 2.5, 0.), (0., 0., 7.))),
    ):
        cell = Crystal.from_matrix(matrix, space_group=group)
        with pytest.warns(RuntimeWarning, match='remain anonymous'):
            result = Crystal.from_cif(cell.to_cif())
        assert result.space_group == group
        assert _read_crystal_record(result) == _read_crystal_record(cell)


def test_rich_native_current_facts_raw_buffers_typed_metadata_and_partial_charge_markers():
    crystal = rich_cell()
    before = captured(crystal)
    with pytest.warns(RuntimeWarning) as findings:
        text = crystal.to_cif()
    assert any('dative' in str(value.message) for value in findings)
    assert any('image code unavailable' in str(value.message) for value in findings)
    result = Crystal.from_cif(text)
    after = captured(result)
    assert_current_equal(before, after)
    assert after.partial_charge_result[0].method == 'fixture'
    assert after.partial_charge_result[1] == ('geometry',)
    assert after.identifier == (True, 'current facts')
    assert after.angle_constraints == ((0, 1, 2),) and after.torsion_constraints == ((0, 1, 2, 3),)
    assert after.pair_wiberg_bond_orders == ((0, 2, .375),)
    metadata = result.molecule.bonds[0].bond_source_metadata
    assert metadata['bytes'] == b'\x00\xff' and metadata['typed'].dtype.str == '>i2'
    np.testing.assert_array_equal(metadata['typed'], [[1, 2], [3, 4]])
    assert type(metadata['scalar']) is np.float32
    assert after.stereo_payload.records[0].group.id == -(2**130)
    assert len(after.stereo_payload.atom_facts) == 2
    np.testing.assert_array_equal(result.matrix, crystal.matrix)
    assert_current_equal(before, captured(crystal))


@pytest.mark.parametrize('representation', ['cell', 'supercell'])
def test_generated_full_history_aliases_maps_and_active_population_roundtrip(representation):
    molecule = Molecule()
    for number, point in zip((6, 8, 8), ((0., 0., 0.), (1., 0., 0.), (-1., 0., 0.))):
        molecule.create_atom(atomic_number=number, coordinates=point, formal_charge=0, implicit_hydrogens=0)
    molecule.add_bond(0, 1, bond_order=1)
    molecule.add_bond(0, 2, bond_order=1)
    group = SpaceGroup(2)
    inversion = SymmUnit.from_xyz('-x,-y,-z', context=group)
    relations = {'equivalent_images': (((0, group.identity(), (0, 0, 0)), (0, inversion, (0, 0, 0))),
                                      ((1, group.identity(), (0, 0, 0)), (2, inversion, (0, 0, 0)))),
                 'occupancies': (.5, .25, .25), 'disorder': (None, ('A', 1), ('A', 1))}
    source = Crystal.from_matrix(np.eye(3) * 5, molecule=molecule, space_group=group, site_relations=relations)
    before = captured(source)
    generated = source.unit_cell() if representation == 'cell' else source.expand((2, 1, 1))
    result = Crystal.from_cif(generated.to_cif())
    assert_current_equal(captured(generated), captured(result))
    assert result.representation == representation and result.space_group == SpaceGroup(1)
    assert _read_crystal_record(result)[6][1] == group
    assert all(len(entry) == 2 for entry in result.image_map)
    assert_current_equal(before, captured(source))


@pytest.mark.parametrize('tag,values', [
    ('_cell_length_a', ('8',)), ('_atom_site_label', ('Edited', 'N2', 'O3', 'H4')),
    ('_atom_site_type_symbol', ('O', 'N', 'O', 'H')), ('_atom_site_fract_x', ('2', '.1', '.2', '.3')),
    ('_atom_site_occupancy', ('1', '?', '?', '?')), ('_atom_site_disorder_group', ('1', '?', '?', '?')),
    ('_geom_bond_atom_site_label_2', ('O3', 'O3', 'H4')),
    ('_geom_bond_hotpot_order', ('2', '1', '1')), ('_geom_bond_hotpot_image', ('[1,0,0]', '[12,-9,2]', '?')),
    ('_space_group_it_number', ('2',)), ('_space_group_symop_operation_xyz', ('x,y,z', '-x,-y,-z')),
])
def test_edited_standard_native_conflicts_reject_instead_of_restoring_hidden_facts(tag, values):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        text = source_cell().to_cif()
        edited = rewrite(text, {tag: values})
        with pytest.raises(ValueError, match='CRYSTAL-') as caught:
            Crystal.from_cif(edited)
    assert caught.value.cif_source == edited


@pytest.mark.parametrize('mutation', ['schema', 'extra', 'record_extra', 'bad_buffer', 'stereo_reference', 'duplicate_labels'])
def test_closed_schema_and_malformed_references_are_rejected(mutation):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        text = rich_cell().to_cif()
    _, tags = _read_cif(text)
    data = json.loads(tags['_hotpot_state_json'][0])
    if mutation == 'schema':
        data['schema'] = 'hotpot.cif/9'
    elif mutation == 'extra':
        data['constructor'] = 'os.system'
    elif mutation == 'record_extra':
        data['record']['call'] = ['str', 'anything']
    elif mutation == 'bad_buffer':
        data['record']['atom_values'] = '00'
    elif mutation == 'stereo_reference':
        data['record']['stereo_payload']['records'][0]['anchor']['atom_indices'] = [999]
    else:
        data['labels'][1] = data['labels'][0]
    edited = rewrite(text, {'_hotpot_state_json': (json.dumps(data),)})
    with pytest.raises(ValueError, match='CRYSTAL-'):
        Crystal.from_cif(edited)


def test_bare_supported_standard_can_be_reimported_after_native_extension_removal():
    cell = source_cell()
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        text = cell.to_cif()
    text = rewrite(text, {'_hotpot_state_json': ()})
    result = Crystal.from_cif(text)
    np.testing.assert_allclose(result.molecule.coordinates, cell.molecule.coordinates, atol=1e-10)
    assert result.bond_images == cell.bond_images
    assert captured(result).charge == (False, None, None)


def test_standalone_and_empty_contained_cell_distinction_and_oriented_precision():
    for molecule in (None, Molecule()):
        cell = Crystal.from_matrix(((0., 5.123456789012345, 0.), (-6., 1.25, 0.), (.5, .75, 7.)), molecule=molecule)
        result = Crystal.from_cif(cell.to_cif())
        assert (result.molecule is None) is (molecule is None)
        np.testing.assert_array_equal(result.matrix, cell.matrix)
        assert result.space_group is None


def test_export_freezes_capture_before_syntax_and_does_not_observe_callback_mutation(monkeypatch):
    cell = source_cell()
    before = captured(cell)
    original = crystal_io._write_cif
    def mutate(name, pairs, loops):
        cell.molecule.atoms[0].coordinates = (99., 98., 97.)
        cell.molecule.bonds[0]._set_bond_metadata(bond_kind='single', bond_source_metadata={'callback': 'late'})
        return original(name, pairs, loops)
    monkeypatch.setattr(crystal_io, '_write_cif', mutate)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        text = cell.to_cif()
    result = Crystal.from_cif(text)
    assert_current_equal(before, captured(result))
    assert tuple(cell.molecule.atoms[0].coordinates) == (99., 98., 97.)
    assert 'callback' not in result.molecule.bonds[0].bond_source_metadata


def test_discovered_nonempty_omissions_are_precise_and_source_is_unchanged():
    cell = source_cell()
    record = captured(cell)
    frame = CoordinateFrame(tuple(range(4)), cell.molecule.coordinates, None, None, _freeze_value({'source': 'fixture'}))
    record = replace(record, source_frames=(frame,))
    cell = Molecule._build_molecule(record).crystal
    cell.molecule._mca_result = object()
    cell.molecule.properties['user_data'] = {'owned': True}
    cell.molecule.runtime_extra = 'extra'
    cell.molecule.atoms[0].custom = 'atom'
    cell.molecule.bonds[0].custom = 'bond'
    with pytest.warns(RuntimeWarning) as findings:
        text = cell.to_cif()
    messages = [str(value.message) for value in findings]
    for fragment in ('source_frames', 'MCA attachment', "properties keys=('user_data',)",
                     "Molecule extras=('runtime_extra',)", "Atom[0] extras=('custom',)", "Bond[0] extras=('custom',)"):
        assert any(fragment in value for value in messages), fragment
    result = Crystal.from_cif(text)
    assert captured(result).source_frames == ()
    assert result.molecule._mca_result is None
    assert result.molecule.properties.keys() == {'cif_source'}
    assert len(cell.molecule._source_frames) == 1 and cell.molecule.properties['user_data'] == {'owned': True}


def test_ordinary_native_cell_export_has_no_false_omission_warning():
    cell = source_cell()
    cell._state = (*cell._state[:3], tuple((bond._uid, (0, 0, 0)) for bond in cell.molecule.bonds), *cell._state[4:])
    with warnings.catch_warnings(record=True) as findings:
        warnings.simplefilter('always')
        cell.to_cif()
    assert findings == []


def test_nonfinite_supplied_export_geometry_rejects_before_syntax():
    cell = source_cell()
    cell.molecule.atoms[0].coordinates = (np.nan, 0., 0.)
    with pytest.raises(ValueError, match='finite supplied Cartesian geometry'):
        cell.to_cif()


def test_optional_dependency_error_is_local_and_math_remains_usable(monkeypatch):
    original = builtins.__import__
    def importing(name, *args, **kwargs):
        if name == 'gemmi' or name.startswith('gemmi.'):
            raise ImportError('isolated optional dependency')
        return original(name, *args, **kwargs)
    monkeypatch.setattr(builtins, '__import__', importing)
    assert Crystal(5, 6, 7, 90, 90, 90).volume == pytest.approx(210)
    for operation in (lambda: Crystal.from_cif(bare()), lambda: source_cell().to_cif()):
        with pytest.raises(ValueError, match='CRYSTAL-UNSUPPORTED-PROJECTION.*optional crystal dependency'):
            with warnings.catch_warnings():
                warnings.simplefilter('ignore', RuntimeWarning)
                operation()


def test_only_gemmi_syntax_api_and_no_external_chemistry_imports(monkeypatch):
    import gemmi
    allowed = ('read_string', 'as_string', 'quote', 'Document')
    syntax = SimpleNamespace(**{name: getattr(gemmi.cif, name) for name in allowed})
    module = ModuleType('gemmi')
    module.cif = syntax
    monkeypatch.setitem(sys.modules, 'gemmi', module)
    original = builtins.__import__
    def importing(name, globals=None, locals=None, fromlist=(), level=0):
        if name.split('.')[0] in ('rdkit', 'openbabel', 'ccdc', 'pymatgen', 'ase'):
            raise AssertionError('forbidden external chemistry import: ' + name)
        if name == 'gemmi' and fromlist != ('cif',):
            raise AssertionError('forbidden Gemmi API: ' + repr(fromlist))
        return original(name, globals, locals, fromlist, level)
    monkeypatch.setattr(builtins, '__import__', importing)
    result = Crystal.from_cif(bare(extra=bonds(['C1 N1 . . S'])))
    with pytest.warns(RuntimeWarning, match='properties keys'):
        text = result.to_cif()
    assert len(Crystal.from_cif(text).molecule.bonds) == 1


def test_shared_stereo_signed_keys_groups_occurrence_order_and_nonatom_slots():
    payload = rich_cell().molecule._capture_factual_record()[0].stereo_payload
    data = _stereo_to_data(payload)
    keys = [-(2**100), 0, 2**120, -3]
    encoded = deepcopy(data)
    for item in encoded['records']:
        item['anchor']['atom_indices'] = [keys[row] for row in item['anchor']['atom_indices']]
        for carrier in item['carriers']:
            if carrier['kind'] == 'atom':
                carrier['atom'] = keys[carrier['atom']]
    for item in encoded['atom_facts']:
        item['atom'] = keys[item['atom']]
    for item in encoded['bond_facts']:
        item['atom_indices'] = [keys[row] for row in item['atom_indices']]
    decoded = _stereo_from_data(encoded, {key: row for row, key in enumerate(keys)}, ((0, 1), (1, 2), (2, 3)))
    assert decoded == payload
    assert decoded.records[0].carriers[1].atom == -17
    assert decoded.records[0].carriers[3].atom == 2
    assert decoded.records[0].group.id == -(2**130)
    assert _stereo_to_data(decoded) == data


@pytest.mark.parametrize('target', ['anchor', 'carrier', 'axis_edge', 'bond_anchor', 'atom_fact', 'bond_fact', 'schema'])
def test_shared_stereo_closure_and_current_graph_references(target):
    data = _stereo_to_data(captured(rich_cell()).stereo_payload)
    edges = ((0, 1), (1, 2), (2, 3))
    if target == 'anchor':
        data['records'][0]['anchor']['atom_indices'] = [99]
    elif target == 'carrier':
        data['records'][0]['carriers'][0]['atom'] = 99
    elif target == 'axis_edge':
        edges = ((0, 1), (1, 2))
    elif target == 'bond_anchor':
        data['records'][1]['anchor']['atom_indices'] = [0, 2]
    elif target == 'atom_fact':
        data['atom_facts'][0]['atom'] = 99
    elif target == 'bond_fact':
        data['bond_facts'][0]['atom_indices'] = [0, 3]
    else:
        data['records'][0]['group']['id'] = '7'
    with pytest.raises((ValueError, TypeError)):
        _stereo_from_data(data, {row: row for row in range(4)}, edges)


@pytest.mark.parametrize('order', [1.25, np.nan])
def test_known_kind_with_nonstandard_or_absent_numeric_order_uses_honest_projection(order):
    cell = source_cell()
    record = captured(cell)
    raw = np.frombuffer(record.bond_values, dtype='<f8').copy()
    raw[0] = order
    cell = Molecule._build_molecule(replace(record, bond_values=raw.tobytes())).crystal
    with pytest.warns(RuntimeWarning, match='cannot express kind single'):
        text = cell.to_cif()
    result = Crystal.from_cif(text)
    assert captured(result).bond_values == captured(cell).bond_values
    assert captured(result).bond_kinds == ('single', 'single', 'single')


def test_nonportable_bond_metadata_rejects_with_affected_row():
    cell = source_cell()
    cell.molecule.bonds[0]._set_bond_metadata(bond_kind='single', bond_source_metadata={'opaque': object()})
    with pytest.raises(ValueError, match=r'CRYSTAL-UNSUPPORTED-PROJECTION: CIF Bond\[0\] metadata'):
        cell.to_cif()


def test_crystal_extras_are_reported_when_discovered():
    cell = Crystal(5, 6, 7, 90, 90, 90)
    cell.user_annotation = 'outside the format'
    with pytest.warns(RuntimeWarning, match="Crystal extras=.*user_annotation"):
        result = Crystal.from_cif(cell.to_cif())
    assert not hasattr(result, 'user_annotation')


@pytest.mark.parametrize('hot_tag,value,reason', [
    ('_geom_bond_hotpot_kind', 'double', 'kinds disagree'),
    ('_geom_bond_hotpot_order', '2', 'orders disagree'),
    ('_geom_bond_hotpot_image', "'[1,0,0]'", 'images disagree'),
])
def test_supplied_hotpot_bond_columns_agree_with_known_standard_facts(hot_tag, value, reason):
    extra = f'loop_\n_geom_bond_atom_site_label_1\n_geom_bond_atom_site_label_2\n_geom_bond_type\n_geom_bond_site_symmetry_2\n{hot_tag}\nC1 N1 S . {value}\n'
    with pytest.raises(ValueError, match=reason):
        Crystal.from_cif(bare(extra=extra))


def test_dual_coordinate_numeric_tolerance_is_geometry_based():
    cell = source_cell()
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        text = cell.to_cif()
    _, tags = _read_cif(text)
    original = list(tags['_atom_site_fract_x'])
    inside = list(original)
    inside[0] = format(float(inside[0]) + 1e-12, '.17g')
    result = Crystal.from_cif(rewrite(text, {'_atom_site_fract_x': inside}))
    assert_current_equal(captured(cell), captured(result))
    outside = list(original)
    outside[0] = format(float(outside[0]) + 1e-7, '.17g')
    with pytest.raises(ValueError, match='coordinates'):
        Crystal.from_cif(rewrite(text, {'_atom_site_fract_x': outside}))


@pytest.mark.parametrize('extra,reason', [
    ('_cell.length_a 5\n', 'repeated normalized CIF tag'),
    ('_symmetry_Int_Tables_number 1\n_space_group_IT_number 2\n', 'conflicting CIF aliases'),
])
def test_duplicate_normalized_tags_and_conflicting_aliases_are_rejected(extra, reason):
    with pytest.raises(ValueError, match=reason):
        Crystal.from_cif(bare(extra=extra))


def test_supplied_incomplete_operator_list_does_not_acquire_a_complete_group():
    # Identity plus a 1/3 translation is not closed without the 2/3 operation.
    with pytest.raises(ValueError, match='CRYSTAL-CONTEXT-INTEGRITY'):
        Crystal.from_cif(bare(ops=OPS + "2 'x+1/3,y,z'\n"))


def test_standard_image_without_explicit_operators_cannot_invent_identity_group():
    with pytest.raises(ValueError, match='unknown CIF Bond symmetry code'):
        Crystal.from_cif(bare(ops='', extra=bonds(['C1 N1 . 1_555 S'])))


@pytest.mark.parametrize('element', [0., 999., 6.5, np.nan, np.inf])
def test_unrepresentable_supplied_native_element_rejects_locally(element):
    cell = source_cell()
    record = captured(cell)
    raw = np.frombuffer(record.atom_values, dtype='<f8').copy()
    raw[FIELD_INDEX['atomic_number']] = element
    cell = Molecule._build_molecule(replace(record, atom_values=raw.tobytes())).crystal
    with pytest.raises(ValueError, match='representable supplied elements'):
        cell.to_cif()
