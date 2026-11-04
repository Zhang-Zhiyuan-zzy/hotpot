"""Plain JSON data for the sole immutable hotpot.crystal/1 record."""

from ._crystal import _RELATION_KEYS, _canonical_crystal_record
from .symmetry import SpaceGroup, SymmUnit


_FIELDS = ('schema', 'matrix', 'group', 'relations', 'bond_images', 'representation',
           'origin_binding', 'image_map', 'bond_origins')
_ORIGIN_FIELDS = ('matrix', 'group', 'sites', 'bonds', 'basis_map')
_GROUP_FIELDS = ('number', 'to_reference', 'operations')


def _fields(data, names):
    if set(data) != set(names):
        raise ValueError('CRYSTAL-CONTEXT-INTEGRITY: portable record requires fields ' + ', '.join(names))
    return tuple(data[name] for name in names)


def _lists(value):
    return [_lists(item) for item in value] if isinstance(value, tuple) else value


def _rational_matrix(matrix):
    return [[str(value) for value in row] for row in matrix]


def _group_to_data(group):
    if group is None:
        return None
    setting = group.setting
    return {'number': group.number,
            'to_reference': None if setting is None else
                [_rational_matrix(setting[1]), [str(value) for value in setting[2]]],
            'operations': [operation.xyz for operation in group.symm_units]}


def _group_from_data(data):
    if data is None:
        return None
    number, setting, operations = _fields(data, _GROUP_FIELDS)
    return SpaceGroup.from_operations(operations, reference_number=number, to_reference=setting)


def _key_to_data(key):
    row, operation, image = key
    return [row, operation.xyz, list(image)]


def _key_from_data(key, group):
    row, symbol, image = key
    # The canonical image-key owner rejects a missing declared group.
    operation = None if group is None else SymmUnit.from_xyz(symbol, context=group)
    return row, operation, image


def _maps_to_data(values):
    return None if values is None else [None if row is None else [_key_to_data(key) for key in row]
                                       for row in values]


def _maps_from_data(values, group):
    return None if values is None else tuple(None if row is None else tuple(_key_from_data(key, group) for key in row)
                                            for row in values)


def _crystal_to_data(record):
    schema, matrix, group, relations, images, representation, origin, image_map, bond_origins = record
    equivalent, occupancies, disorder = relations
    binding = None
    if origin is not None:
        source_matrix, source_group, sites, bonds, basis_map = origin
        binding = {'matrix': _lists(source_matrix), 'group': _group_to_data(source_group),
                   'sites': _lists(sites), 'bonds': _lists(bonds),
                   'basis_map': None if basis_map is None else _rational_matrix(basis_map)}
    return {'schema': schema, 'matrix': _lists(matrix), 'group': _group_to_data(group),
            'relations': {'equivalent_images': [[_key_to_data(key) for key in entry] for entry in equivalent],
                          'occupancies': list(occupancies), 'disorder': _lists(disorder)},
            'bond_images': _lists(images), 'representation': representation, 'origin_binding': binding,
            'image_map': _maps_to_data(image_map), 'bond_origins': _maps_to_data(bond_origins)}


def _crystal_from_data(data, atom_count, bond_count):
    schema, matrix, group, relations, images, representation, origin, image_map, bond_origins = _fields(data, _FIELDS)
    group = _group_from_data(group)
    equivalent, occupancies, disorder = _fields(relations, _RELATION_KEYS)
    relations = (tuple(tuple(_key_from_data(key, group) for key in entry) for entry in equivalent),
                 occupancies, disorder)
    source_group = None
    if origin is not None:
        source_matrix, source_group, sites, bonds, basis_map = _fields(origin, _ORIGIN_FIELDS)
        source_group = _group_from_data(source_group)
        origin = source_matrix, source_group, sites, bonds, basis_map
    return _canonical_crystal_record((schema, matrix, group, relations, images, representation, origin,
        _maps_from_data(image_map, source_group), _maps_from_data(bond_origins, source_group)), atom_count, bond_count)
