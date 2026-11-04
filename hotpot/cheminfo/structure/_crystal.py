"""One closed Crystal record and member binding; no chemistry or external toolkit."""

from collections.abc import Mapping
from contextlib import nullcontext
import operator
from types import MappingProxyType

import numpy as np

from .symmetry import SpaceGroup, SymmUnit
from ..rules.mathematics.cell import _cell_matrix, _validate_matrix, _check_metric
from ..rules.mathematics import symmetry as _affine

_SCHEMA = 'hotpot.crystal/1'
_RELATION_KEYS = ('equivalent_images', 'occupancies', 'disorder')


def _image(value):
    value = tuple(map(operator.index, value))
    if len(value) != 3:
        raise ValueError('CRYSTAL-IMAGE-MAP-INTEGRITY: image needs three integers')
    return value


def _basis(value):
    return tuple(tuple(map(float, row)) for row in _validate_matrix(value))


def _group_metric(matrix, group):
    if group is not None:
        if not isinstance(group, SpaceGroup):
            raise TypeError('CRYSTAL-CONTEXT-INTEGRITY: space_group requires SpaceGroup or None')
        _check_metric(matrix, group.point_rotations)


def _image_key(value, count, group):
    row, operation, image = value
    row = operator.index(row)
    if not 0 <= row < count:
        raise ValueError('CRYSTAL-IMAGE-MAP-INTEGRITY: source Atom row outside its frame')
    if group is None or not isinstance(operation, SymmUnit) or operation.context != group:
        raise ValueError('CRYSTAL-IMAGE-MAP-INTEGRITY: operation needs the declared source group')
    operation = SymmUnit.from_affine(operation.R, operation.t, context=group)
    return row, operation, _image(image)


def _relations(value, count, group):
    if value is None:
        return (), (None,) * count, (None,) * count
    if isinstance(value, Mapping):
        if set(value) != set(_RELATION_KEYS):
            raise ValueError('CRYSTAL-IMAGE-MAP-INTEGRITY: site_relations requires equivalent_images, occupancies and disorder')
        value = tuple(value[key] for key in _RELATION_KEYS)
    equivalents, occupancies, disorder = value
    equivalents = tuple(tuple(_image_key(key, count, group) for key in entry) for entry in equivalents)
    if any(len(entry) < 2 or len(set(entry)) != len(entry) for entry in equivalents):
        raise ValueError('CRYSTAL-IMAGE-MAP-INTEGRITY: equivalence groups need distinct image keys')
    occupancies = tuple(None if item is None else float(item) for item in occupancies)
    disorder = tuple(None if item is None else tuple(item) for item in disorder)
    if len(occupancies) != count or len(disorder) != count:
        raise ValueError('CRYSTAL-IMAGE-MAP-INTEGRITY: site values must align with Atom membership')
    if any(item is not None and (len(item) != 2 or any(
            value is not None and type(value) not in (str, int) for value in item)) for item in disorder):
        raise ValueError('CRYSTAL-IMAGE-MAP-INTEGRITY: disorder needs optional assembly/group pairs')
    return equivalents, occupancies, disorder


def _historical(value):
    """The finite historical descriptor domain, without mutable object capture."""
    if value is None or type(value) in (str, bool, int, float):
        return value
    if isinstance(value, np.generic):
        return _historical(value.item())
    if type(value) in (tuple, list):
        return tuple(_historical(item) for item in value)
    raise TypeError('CRYSTAL-IMAGE-MAP-INTEGRITY: historical descriptors require primitive values')


def _origin(value):
    if value is None:
        return None
    matrix, group, sites, bonds, frame = value
    matrix = _basis(matrix)
    _group_metric(matrix, group)
    if group is None:
        raise ValueError('CRYSTAL-IMAGE-MAP-INTEGRITY: generated origins need a known source group')
    if frame is not None:
        frame = _affine._matrix(frame)
        _affine._inverse(frame)
    return matrix, group, tuple(map(_historical, sites)), tuple(map(_historical, bonds)), frame


def _origin_maps(values, count, source_count, group):
    values = tuple(None if row is None else tuple(_image_key(key, source_count, group) for key in row)
                   for row in values)
    if len(values) != count or any(row == () for row in values):
        raise ValueError('CRYSTAL-IMAGE-MAP-INTEGRITY: origins must align and retain every contributing key')
    return values


def _canonical_crystal_record(value, atom_count, bond_count):
    value = tuple(value)
    if len(value) != 9 or value[0] != _SCHEMA:
        raise ValueError('CRYSTAL-CONTEXT-INTEGRITY: current hotpot.crystal/1 record required')
    _, matrix, group, relations, images, representation, origin, image_map, bond_origins = value
    matrix = _basis(matrix)
    _group_metric(matrix, group)
    relations = _relations(relations, atom_count, group)
    images = tuple(None if image is None else _image(image) for image in images)
    if len(images) != bond_count:
        raise ValueError('CRYSTAL-IMAGE-MAP-INTEGRITY: Bond images must align with complete Bond order')
    if representation not in ('independent', 'cell', 'supercell'):
        raise ValueError('CRYSTAL-CONTEXT-INTEGRITY: unknown content representation')
    origin = _origin(origin)
    if representation == 'independent':
        if origin is not None or image_map is not None or bond_origins is not None:
            raise ValueError('CRYSTAL-IMAGE-MAP-INTEGRITY: independent content has no generated origin maps')
    else:
        if group != SpaceGroup(1):
            raise ValueError('CRYSTAL-CONTEXT-INTEGRITY: generated cell/supercell requires explicit active P1')
        if origin is None or image_map is None or bond_origins is None:
            raise ValueError('CRYSTAL-IMAGE-MAP-INTEGRITY: generated content requires its source binding and maps')
        if origin[4] is not None and not np.allclose(
                np.asarray(origin[0]).T @ np.asarray(origin[4], dtype=float), np.asarray(matrix).T,
                rtol=1e-10, atol=1e-10):
            raise ValueError('CRYSTAL-IMAGE-MAP-INTEGRITY: exact historical basis map disagrees with active cell')
        image_map = _origin_maps(image_map, atom_count, len(origin[2]), origin[1])
        bond_origins = _origin_maps(bond_origins, bond_count, len(origin[3]), origin[1])
    return _SCHEMA, matrix, group, relations, images, representation, origin, image_map, bond_origins


def _record_from_parameters(parameters, atom_count=0, bond_count=0, *, unknown_images=False):
    return _canonical_crystal_record((_SCHEMA, _basis(_cell_matrix(parameters)), None,
        None, (None if unknown_images else (0, 0, 0),) * bond_count,
        'independent', None, None, None), atom_count, bond_count)


def _state_from_record(record, atom_uids, bond_uids):
    _, matrix, group, relations, images, representation, origin, image_map, bond_origins = record
    equivalents, occupancies, disorder = relations
    equivalents = tuple(tuple((atom_uids[row], operation, image) for row, operation, image in entry)
                        for entry in equivalents)
    relations = (equivalents, tuple(zip(atom_uids, occupancies)), tuple(zip(atom_uids, disorder)))
    return (matrix, group, relations, tuple(zip(bond_uids, images)), representation, origin,
            None if image_map is None else tuple(zip(atom_uids, image_map)),
            None if bond_origins is None else tuple(zip(bond_uids, bond_origins)))


def _capture_crystal_record(crystal, atoms=(), bonds=()):
    matrix, group, relations, images, representation, origin, image_map, bond_origins = crystal._state
    rows = {atom._uid: row for row, atom in enumerate(atoms)}
    equivalents, occupancies, disorder = relations
    selected = []
    for entry in equivalents:
        kept = tuple((rows[uid], operation, image) for uid, operation, image in entry if uid in rows)
        if len(kept) >= 2:
            selected.append(kept)
    occupancies, disorder, images = dict(occupancies), dict(disorder), dict(images)
    source_map = None if image_map is None else dict(image_map)
    bond_map = None if bond_origins is None else dict(bond_origins)
    return (_SCHEMA, matrix, group,
            (tuple(selected), tuple(occupancies.get(atom._uid) for atom in atoms),
             tuple(disorder.get(atom._uid) for atom in atoms)),
            tuple(images.get(bond._uid, (0, 0, 0)) for bond in bonds), representation, origin,
            None if source_map is None else tuple(source_map.get(atom._uid) for atom in atoms),
            None if bond_map is None else tuple(bond_map.get(bond._uid) for bond in bonds))


def _read_crystal_record(crystal):
    mol = crystal._molecule
    with nullcontext() if mol is None else mol._mutation_lock:
        active = mol is not None and mol._crystal is crystal
        return _capture_crystal_record(crystal, mol._atoms if active else (), mol._bonds if active else ())


def _public_relations(crystal):
    relations = _read_crystal_record(crystal)[3]
    return MappingProxyType(dict(zip(_RELATION_KEYS, relations)))


def _new_crystal(record, molecule=None, *, crystal_type=None):
    from .crystal import Crystal
    crystal_type = Crystal if crystal_type is None else crystal_type
    value = object.__new__(crystal_type)
    value._molecule = molecule
    atom_uids = () if molecule is None else tuple(atom._uid for atom in molecule._atoms)
    bond_uids = () if molecule is None else tuple(bond._uid for bond in molecule._bonds)
    record = _canonical_crystal_record(record, len(atom_uids), len(bond_uids))
    value._state = _state_from_record(record, atom_uids, bond_uids)
    return value


def _detach_crystal(crystal):
    matrix, group = crystal._state[:2]
    crystal._molecule = None
    crystal._state = (matrix, group, ((), (), ()), (), 'independent', None, None, None)


def _install_crystal_record(mol, record):
    from ._commit import _current_transaction
    transaction = _current_transaction(mol)
    previous = mol._crystal
    if transaction is not None:
        transaction.touch_crystal(previous)
    value = None if record is None else _new_crystal(record, mol)
    if transaction is not None and value is not None:
        transaction.touch_crystal(value, new=True)
    if previous is not None:
        _detach_crystal(previous)
    mol._crystal = value
    return value


def _prune_crystal_members(mol):
    crystal = mol._crystal
    if crystal is None:
        return
    from ._commit import _current_transaction
    transaction = _current_transaction(mol)
    if transaction is not None:
        transaction.touch_crystal(crystal)
    record = _capture_crystal_record(crystal, mol._atoms, mol._bonds)
    crystal._state = _state_from_record(record, tuple(atom._uid for atom in mol._atoms),
                                       tuple(bond._uid for bond in mol._bonds))


def _merge_crystal_records(left, right, left_count, right_count):
    if left is None and right is None:
        return None
    if left is None:
        if left_count:
            raise ValueError('CRYSTAL-CONTEXT-INTEGRITY: nonempty components require a shared explicit cell context')
        return right
    if right is None:
        if right_count:
            raise ValueError('CRYSTAL-CONTEXT-INTEGRITY: nonempty components require a shared explicit cell context')
        return left
    if (left[1], left[2], left[5], left[6]) != (right[1], right[2], right[5], right[6]):
        raise ValueError('CRYSTAL-CONTEXT-INTEGRITY: components require compatible unit cells, groups and origin frames')
    le, lo, ld = left[3]
    re, ro, rd = right[3]
    shifted = tuple(tuple((row + left_count, operation, image) for row, operation, image in entry)
                    for entry in re)
    return (_SCHEMA, left[1], left[2], (le + shifted, lo + ro, ld + rd), left[4] + right[4],
            left[5], left[6], None if left[7] is None else left[7] + right[7],
            None if left[8] is None else left[8] + right[8])


def _restore_standalone_crystal(crystal_type, record):
    return _new_crystal(record, crystal_type=crystal_type)


def _restore_active_crystal(molecule):
    return molecule._crystal


def _import_crystal_context(record, atom_count, bond_count):
    """Retain an explicitly preconfigured empty context, with unknown imported images."""
    if record is None:
        return None
    return _canonical_crystal_record((_SCHEMA, record[1], record[2], None,
        (None,) * bond_count, 'independent', None, None, None), atom_count, bond_count)
