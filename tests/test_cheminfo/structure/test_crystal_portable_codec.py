"""Portable Crystal records retain exact groups, image frames and history."""

import builtins
from copy import deepcopy
from fractions import Fraction as Q
import json

import numpy as np
import pytest

from hotpot import Crystal, Molecule, SpaceGroup, SymmUnit
from hotpot.cheminfo.structure._crystal import _canonical_crystal_record, _read_crystal_record
from hotpot.cheminfo.structure._crystal_codec import (
    _crystal_from_data, _crystal_to_data, _group_from_data, _group_to_data,
)


I = ((1, 0, 0), (0, 1, 0), (0, 0, 1))
ZERO = (0, 0, 0)


def reference_data():
    """Independent P2_1 fixture, including an oriented basis and full carries."""
    return {
        'schema': 'hotpot.crystal/1',
        'matrix': [[0., 5., 0.], [-6., 0., 0.], [0., 0., 7.]],
        'group': {'number': 4, 'to_reference': [[['1', '0', '0'], ['0', '1', '0'], ['0', '0', '1']],
                                                ['0', '0', '0']],
                  'operations': ['x,y,z', '-x,y+1/2,-z']},
        'relations': {'equivalent_images': [[[0, 'x,y,z', [2, -3, 1]],
                                             [1, '-x+3,y-5/2,-z+2', [-1, 4, 0]]]],
                      'occupancies': [.5, None], 'disorder': [['A', 1], None]},
        'bond_images': [[2, -1, 0]],
        'representation': 'independent',
        'origin_binding': None,
        'image_map': None,
        'bond_origins': None,
    }


def source_crystal():
    value = Molecule()
    for number, point in zip((6, 7), ((-.625, 1.5, .875), (5.625, 2.25, 1.75))):
        value.create_atom(atomic_number=number, coordinates=point, formal_charge=0, implicit_hydrogens=0)
    value.add_bond(0, 1, bond_order=1)
    return Crystal.from_matrix(np.diag((5., 6., 7.)), molecule=value,
                               space_group=SpaceGroup(4), bond_images=((2, -1, 0),))


def historical_record():
    group = SpaceGroup(4)
    operation = SymmUnit.from_xyz('-x+5,y-7/2,-z-2', context=group)
    basis = ((Q(1, 2), Q(1, 3), Q(0)), (Q(0), Q(1), Q(0)), (Q(0), Q(0), Q(1)))
    origin = (((6., 0., 0.), (0., 10., 0.), (0., 0., 14.)), group,
              (('site', ('nested', 7, True, None, 1.25)),), (), basis)
    return _canonical_crystal_record(('hotpot.crystal/1',
        ((3., 0., 0.), (2., 10., 0.), (0., 0., 14.)), SpaceGroup(1),
        ((), (None,), (None,)), (), 'cell', origin,
        (((0, operation, (2, -4, 9)),),), ()), 1, 0)


def json_data(value):
    if type(value) is dict:
        assert all(type(key) is str for key in value)
        for item in value.values():
            json_data(item)
    elif type(value) is list:
        for item in value:
            json_data(item)
    else:
        assert value is None or type(value) in (str, bool, int, float)


def test_independent_reference_fixture_preserves_exact_values_and_operation_context():
    data = reference_data()
    record = _crystal_from_data(data, 2, 1)
    group = record[2]
    assert record[:2] == ('hotpot.crystal/1', ((0., 5., 0.), (-6., 0., 0.), (0., 0., 7.)))
    assert group == SpaceGroup(4) and group.setting == (4, I, ZERO)
    assert record[3][1:] == ((.5, None), (('A', 1), None))
    first, second = record[3][0][0]
    assert first == (0, group.identity(), (2, -3, 1))
    assert second[0] == 1 and second[1].R == ((-1, 0, 0), (0, 1, 0), (0, 0, -1))
    assert second[1].t == (3, Q(-5, 2), 2) and second[2] == (-1, 4, 0)
    assert first[1].context is group and second[1].context is group
    assert record[4:] == (((2, -1, 0),), 'independent', None, None, None)
    assert _crystal_to_data(record) == data


@pytest.mark.parametrize('number,count', ((1, 1), (2, 2), (4, 2), (5, 4), (19, 4), (75, 4), (146, 9)))
def test_reference_group_helpers_retain_validated_complete_tables(number, count):
    group = SpaceGroup(number)
    data = _group_to_data(group)
    assert len(data['operations']) == count
    assert data['to_reference'] == [[['1', '0', '0'], ['0', '1', '0'], ['0', '0', '1']], ['0', '0', '0']]
    assert _group_from_data(json.loads(json.dumps(data))) == group


def test_anonymous_group_does_not_acquire_a_reference_identity():
    group = SpaceGroup.from_operations(('x,y,z', '-x+2/7,-y-4/5,-z+6/11'))
    data = reference_data()
    data['group'] = _group_to_data(group)
    data['relations']['equivalent_images'] = []
    record = _crystal_from_data(json.loads(json.dumps(data)), 2, 1)
    assert record[2] == group
    assert record[2].number is None and record[2].setting is None and record[2].hall is None
    assert _crystal_to_data(record)['group']['number'] is None
    assert _crystal_to_data(record)['group']['to_reference'] is None


def test_changed_reference_setting_preserves_rational_origin_and_basis():
    basis = ((0, 1, 0), (1, 0, 0), (0, 0, -1))
    origin = (Q(1, 7), Q(-2, 5), Q(3, 11))
    group = SpaceGroup(4).change_setting(basis, origin)
    crystal = Crystal.from_matrix(np.array(basis).T @ np.diag((5., 6., 7.)), space_group=group)
    record = _read_crystal_record(crystal)
    data = _crystal_to_data(record)
    assert data['group']['to_reference'] == [[['0', '1', '0'], ['1', '0', '0'], ['0', '0', '-1']],
                                               ['1/7', '-2/5', '3/11']]
    restored = _crystal_from_data(json.loads(json.dumps(data)), 0, 0)
    assert restored == record and restored[2].setting == (4, basis, origin)
    assert restored[2].hall is None and restored[2].hermann_mauguin is None


def test_centered_to_primitive_setting_retains_changed_translation_index():
    basis = ((Q(1, 2), Q(-1, 2), 0), (Q(1, 2), Q(1, 2), 0), (0, 0, 1))
    group = SpaceGroup(5).change_setting(basis, ZERO)
    assert group.cell_order == 2 and SpaceGroup(5).cell_order == 4
    data = _group_to_data(group)
    assert data['to_reference'][0] == [['1/2', '-1/2', '0'], ['1/2', '1/2', '0'], ['0', '0', '1']]
    assert _group_from_data(json.loads(json.dumps(data))) == group


@pytest.mark.parametrize('kind', ('independent', 'cell', 'supercell'))
def test_real_generated_records_roundtrip_without_changing_source_facts(kind):
    source = source_crystal()
    before = source.molecule._capture_factual_record()[0]
    crystal = source if kind == 'independent' else source.unit_cell() if kind == 'cell' else source.expand((2, 1, 1))
    record = _read_crystal_record(crystal)
    data = _crystal_to_data(record)
    json_data(data)
    restored = _crystal_from_data(json.loads(json.dumps(data, allow_nan=False)),
                                  len(crystal.molecule.atoms), len(crystal.molecule.bonds))
    assert restored == record
    assert source.molecule._capture_factual_record()[0].cell == before.cell
    assert source.molecule._capture_factual_record()[0].atom_values == before.atom_values
    if kind != 'independent':
        assert data['group']['number'] == 1 and data['origin_binding']['group']['number'] == 4
        historical_group = restored[6][1]
        keys = [key for row in restored[7] for key in row]
        assert any(key[1].R != I for key in keys)
        assert all(key[1].context is historical_group for key in keys)
        assert all(key[1].context is historical_group for row in restored[8] for key in row)
        assert data['image_map'][0][0][0] == 0


def test_generated_identification_retains_all_site_and_bond_origin_aliases():
    molecule = Molecule()
    for number, point in zip((6, 8, 8), ((0., 0., 0.), (1., 0., 0.), (-1., 0., 0.))):
        molecule.create_atom(atomic_number=number, coordinates=point, formal_charge=0, implicit_hydrogens=0)
    molecule.add_bond(0, 1, bond_order=1)
    molecule.add_bond(0, 2, bond_order=1)
    group = SpaceGroup(2)
    inversion = SymmUnit.from_xyz('-x,-y,-z', context=group)
    relations = {'equivalent_images': (((0, group.identity(), ZERO), (0, inversion, ZERO)),
                                      ((1, group.identity(), ZERO), (2, inversion, ZERO))),
                 'occupancies': (None,) * 3, 'disorder': (None,) * 3}
    source = Crystal.from_matrix(np.eye(3) * 5, molecule=molecule, space_group=group, site_relations=relations)
    record = _read_crystal_record(source.unit_cell())
    assert all(len(entry) == 2 for entry in record[7])
    assert all(len(entry) == 2 for entry in record[8])
    restored = _crystal_from_data(json.loads(json.dumps(_crystal_to_data(record))), 3, 2)
    assert restored == record


def test_historical_rational_basis_full_operation_and_integer_image_remain_distinct():
    record = historical_record()
    data = _crystal_to_data(record)
    assert data['origin_binding']['basis_map'] == [['1/2', '1/3', '0'], ['0', '1', '0'], ['0', '0', '1']]
    assert data['image_map'] == [[[0, '-x+5,y-7/2,-z-2', [2, -4, 9]]]]
    json_data(data)
    restored = _crystal_from_data(json.loads(json.dumps(data, allow_nan=False)), 1, 0)
    assert restored == record
    assert restored[6][2] == (('site', ('nested', 7, True, None, 1.25)),)
    assert all(type(value) is Q for row in restored[6][4] for value in row)
    assert restored[7][0][0][1].t == (5, Q(-7, 2), -2)
    assert restored[7][0][0][2] == (2, -4, 9)


def test_unknown_group_stays_unknown_with_unknown_bond_images():
    data = reference_data()
    data['group'] = None
    data['relations']['equivalent_images'] = []
    data['bond_images'] = [None]
    record = _crystal_from_data(data, 2, 1)
    assert record[2] is None and record[4] == (None,)
    assert _crystal_to_data(record)['group'] is None
    assert _group_to_data(None) is None and _group_from_data(None) is None


def test_unknown_generated_origins_and_basis_map_are_preserved():
    record = historical_record()
    unknown = _canonical_crystal_record((*record[:6], (*record[6][:4], None), (None,), ()), 1, 0)
    data = _crystal_to_data(unknown)
    assert data['origin_binding']['basis_map'] is None and data['image_map'] == [None]
    assert _crystal_from_data(json.loads(json.dumps(data)), 1, 0) == unknown


def test_encoded_outputs_do_not_share_mutable_state_with_each_other_or_record():
    record = historical_record()
    first, second = _crystal_to_data(record), _crystal_to_data(record)
    expected = deepcopy(second)
    first['matrix'][0][0] = 999.
    first['group']['operations'].append('-x,-y,-z')
    first['relations']['occupancies'][0] = .25
    first['origin_binding']['group']['to_reference'][0][0][0] = '999'
    first['origin_binding']['sites'][0][1].append('changed')
    first['origin_binding']['basis_map'][0][0] = '999'
    first['image_map'][0][0][2][0] = 999
    assert second == expected and _crystal_to_data(record) == expected


def test_decoded_record_does_not_retain_mutable_input_containers():
    data = _crystal_to_data(historical_record())
    record = _crystal_from_data(data, 1, 0)
    expected = _crystal_to_data(record)
    data['matrix'][0][0] = 999.
    data['origin_binding']['sites'][0][1].append('changed')
    data['origin_binding']['matrix'][0][0] = 999.
    data['origin_binding']['basis_map'][0][0] = '999'
    data['origin_binding']['group']['operations'].clear()
    data['image_map'][0][0][2][0] = 999
    assert _crystal_to_data(record) == expected
    independent = reference_data()
    record = _crystal_from_data(independent, 2, 1)
    expected = _crystal_to_data(record)
    independent['relations']['disorder'][0].append('changed')
    independent['relations']['equivalent_images'][0][0][2][0] = 999
    independent['relations']['occupancies'][0] = 0.
    independent['bond_images'][0][0] = 999
    assert _crystal_to_data(record) == expected


@pytest.mark.parametrize('field', tuple(reference_data()))
def test_missing_top_level_field_rejects_without_defaulting(field):
    data = reference_data()
    del data[field]
    with pytest.raises(ValueError, match='CRYSTAL-CONTEXT-INTEGRITY: portable record requires fields'):
        _crystal_from_data(data, 2, 1)


@pytest.mark.parametrize('section,field', (('group', 'number'), ('group', 'to_reference'), ('group', 'operations'),
                                         ('relations', 'equivalent_images'), ('relations', 'occupancies'),
                                         ('relations', 'disorder')))
def test_missing_group_or_relation_field_rejects(section, field):
    data = reference_data()
    del data[section][field]
    with pytest.raises(ValueError, match='CRYSTAL-CONTEXT-INTEGRITY: portable record requires fields'):
        _crystal_from_data(data, 2, 1)


@pytest.mark.parametrize('field', ('matrix', 'group', 'sites', 'bonds', 'basis_map'))
def test_missing_historical_binding_field_rejects(field):
    data = _crystal_to_data(historical_record())
    del data['origin_binding'][field]
    with pytest.raises(ValueError, match='CRYSTAL-CONTEXT-INTEGRITY: portable record requires fields'):
        _crystal_from_data(data, 1, 0)


@pytest.mark.parametrize('section', (None, 'group', 'relations', 'origin_binding'))
def test_unknown_fields_reject_instead_of_silently_disappearing(section):
    data = _crystal_to_data(historical_record())
    target = data if section is None else data[section]
    target['unrecognized'] = 'would otherwise be lost'
    with pytest.raises(ValueError, match='CRYSTAL-CONTEXT-INTEGRITY: portable record requires fields'):
        _crystal_from_data(data, 1, 0)


@pytest.mark.parametrize('number,setting', ((None, [[['1', '0', '0'], ['0', '1', '0'], ['0', '0', '1']], ['0', '0', '0']]),
                                           (4, None)))
def test_reference_number_and_setting_remain_paired(number, setting):
    data = reference_data()
    data['group']['number'], data['group']['to_reference'] = number, setting
    with pytest.raises(ValueError, match='reference number and setting map are paired'):
        _crystal_from_data(data, 2, 1)


@pytest.mark.parametrize('kind', ('reference', 'setting', 'incomplete', 'float_setting', 'executable_symbol'))
def test_corrupt_group_claims_reject_through_the_exact_group_owner(kind):
    data = reference_data()
    group = data['group']
    if kind == 'reference':
        group['number'] = 2
    elif kind == 'setting':
        group['to_reference'][1][0] = '1/7'
    elif kind == 'incomplete':
        group['operations'] = ['x,y,z', '-y,x,z']
    elif kind == 'float_setting':
        group['to_reference'][0][0][0] = 1.
    else:
        group['operations'] = ['__import__("os").system("invalid")']
    with pytest.raises((ValueError, TypeError), match='CRYSTAL-CONTEXT-INTEGRITY'):
        _crystal_from_data(data, 2, 1)


@pytest.mark.parametrize('kind', ('schema', 'singular', 'metric', 'row', 'operation', 'image', 'bond_count',
                                 'occupancies', 'disorder', 'duplicate_keys', 'unknown_group'))
def test_corrupt_independent_domain_rejects_through_existing_owners(kind):
    data = reference_data()
    if kind == 'schema':
        data['schema'] = 'hotpot.crystal/0'
    elif kind == 'singular':
        data['matrix'][1] = data['matrix'][0][:]
    elif kind == 'metric':
        data['matrix'][1][1] = .25
    elif kind == 'row':
        data['relations']['equivalent_images'][0][0][0] = 2
    elif kind == 'operation':
        data['relations']['equivalent_images'][0][0][1] = '-x,-y,-z'
    elif kind == 'image':
        data['relations']['equivalent_images'][0][0][2] = [1, 2]
    elif kind == 'bond_count':
        data['bond_images'] = []
    elif kind == 'occupancies':
        data['relations']['occupancies'] = []
    elif kind == 'disorder':
        data['relations']['disorder'][0] = ['A', ['nested']]
    elif kind == 'duplicate_keys':
        data['relations']['equivalent_images'][0][1] = deepcopy(data['relations']['equivalent_images'][0][0])
    else:
        data['group'] = None
    with pytest.raises(ValueError, match='CRYSTAL-(CONTEXT|IMAGE-MAP)-INTEGRITY'):
        _crystal_from_data(data, 2, 1)


@pytest.mark.parametrize('kind', ('source_row', 'empty_origins', 'map_count', 'bond_map_count', 'basis_map',
                                 'source_group', 'source_metric', 'history', 'active_group', 'representation'))
def test_corrupt_generated_binding_rejects_through_the_canonical_record_owner(kind):
    data = _crystal_to_data(historical_record())
    if kind == 'source_row':
        data['image_map'][0][0][0] = 1
    elif kind == 'empty_origins':
        data['image_map'][0] = []
    elif kind == 'map_count':
        data['image_map'] = []
    elif kind == 'bond_map_count':
        data['bond_origins'] = [None]
    elif kind == 'basis_map':
        data['origin_binding']['basis_map'][0][0] = '1'
    elif kind == 'source_group':
        data['origin_binding']['group'] = None
    elif kind == 'source_metric':
        data['origin_binding']['matrix'][1][0] = .25
    elif kind == 'history':
        data['origin_binding']['sites'][0] = {'opaque': 'unsupported'}
    elif kind == 'active_group':
        data['group'] = _group_to_data(SpaceGroup(4))
    else:
        data['representation'] = 'independent'
    with pytest.raises((ValueError, TypeError), match='CRYSTAL-(CONTEXT|IMAGE-MAP)-INTEGRITY'):
        _crystal_from_data(data, 1, 0)


def test_portable_codec_requires_no_optional_toolkit_or_pickle(monkeypatch):
    original = builtins.__import__
    def guarded(name, *args, **kwargs):
        if name.split('.')[0] in ('rdkit', 'openbabel', 'ccdc', 'gemmi', 'pybel', 'pickle'):
            raise AssertionError('Forbidden portable Crystal dependency: ' + name)
        return original(name, *args, **kwargs)
    monkeypatch.setattr(builtins, '__import__', guarded)
    data = reference_data()
    assert _crystal_to_data(_crystal_from_data(data, 2, 1)) == data
    generated = _read_crystal_record(source_crystal().unit_cell())
    assert _crystal_from_data(json.loads(json.dumps(_crystal_to_data(generated))), 4, 2) == generated
