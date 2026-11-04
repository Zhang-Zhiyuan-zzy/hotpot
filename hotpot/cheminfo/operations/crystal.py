"""Complete Crystal fact publication and explicit Cartesian/cell operations."""

from contextlib import contextmanager, nullcontext

from ..rules.mathematics.cell import (
    _cell_matrix, _matrix_to_params, _replace_parameter, _to_cartesian, _to_fractional,
)
from ..structure._commit import _FactTransaction
from ..structure._crystal import (
    _SCHEMA, _basis, _canonical_crystal_record, _capture_crystal_record,
    _detach_crystal, _install_crystal_record, _read_crystal_record, _state_from_record,
)
from ..structure._fields import ATOM_SOURCE_CODES, FIELD_INDEX, FIELD_PRESENCE_BITS
from ..structure._results import _invalidate_results
from ..structure.symmetry import SymmUnit

__all__ = ()


@contextmanager
def _scope(crystal):
    molecule = crystal._molecule
    if molecule is None:
        yield None
        return
    with _FactTransaction(molecule, producer=ATOM_SOURCE_CODES['explicit']) as transaction:
        if crystal._molecule is not molecule or molecule._crystal is not crystal:
            raise RuntimeError('CRYSTAL-CONTEXT-INTEGRITY: active association changed while acquiring the Molecule lock')
        yield transaction


def _coordinates(molecule):
    mask = sum(FIELD_PRESENCE_BITS[FIELD_INDEX[name]] for name in ('x', 'y', 'z'))
    if not ((molecule._atom_table.presence() & mask) == mask).all():
        raise ValueError('CRYSTAL-CONTEXT-INTEGRITY: complete supplied Cartesian coordinates required')
    return molecule.coordinates


def _record(matrix, group, relations, images, atom_count, bond_count):
    images = ((0, 0, 0),) * bond_count if images is None else images
    return _canonical_crystal_record((_SCHEMA, matrix, group, relations, images,
                                      'independent', None, None, None), atom_count, bond_count)


def _initialize_crystal(crystal, matrix, molecule, group, relations, images):
    scope = (nullcontext(None) if molecule is None else
             _FactTransaction(molecule, producer=ATOM_SOURCE_CODES['explicit']))
    with scope as transaction:
        if molecule is not None and molecule._crystal is not None:
            raise ValueError('CRYSTAL-CONTEXT-INTEGRITY: Molecule already has an active Crystal; use create_crystal')
        atoms, bonds = ((), ()) if molecule is None else (molecule._atoms, molecule._bonds)
        record = _record(matrix, group, relations, images, len(atoms), len(bonds))
        crystal._state = _state_from_record(record, tuple(atom._uid for atom in atoms),
                                            tuple(bond._uid for bond in bonds))
        crystal._molecule = None
        if molecule is not None:
            transaction.touch_crystal()
            transaction.touch_crystal(crystal, new=True)
            _invalidate_results(molecule, ('cell',), transaction)
            crystal._molecule, molecule._crystal = molecule, crystal


def _create_crystal(molecule, parameters, group, relations, images):
    with _FactTransaction(molecule, producer=ATOM_SOURCE_CODES['explicit']) as transaction:
        record = _record(_cell_matrix(parameters), group, relations, images,
                         len(molecule._atoms), len(molecule._bonds))
        previous = molecule._crystal
        if previous is not None and _read_crystal_record(previous) == record:
            return previous
        if previous is None or previous._state[0] != record[1]:
            _invalidate_results(molecule, ('cell',), transaction)
        return _install_crystal_record(molecule, record)


def _publish(crystal, record, transaction):
    molecule = crystal._molecule
    atoms, bonds = ((), ()) if molecule is None else (molecule._atoms, molecule._bonds)
    if transaction is not None:
        transaction.touch_crystal(crystal)
        if crystal._state[0] != record[1]:
            _invalidate_results(molecule, ('cell',), transaction)
    crystal._state = _state_from_record(record, tuple(atom._uid for atom in atoms),
                                        tuple(bond._uid for bond in bonds))


def _set_cell_locked(crystal, matrix, preserve, transaction):
    if preserve not in ('cartesian', 'fractional'):
        raise ValueError("CRYSTAL-CONTEXT-INTEGRITY: preserve requires 'cartesian' or 'fractional'")
    old = _read_crystal_record(crystal)
    origin = old[6]
    if origin is not None and _basis(matrix) != old[1]:
        origin = (*origin[:4], None)
    candidate = _canonical_crystal_record((*old[:1], matrix, *old[2:6], origin, *old[7:]),
                                           len(old[3][1]), len(old[4]))
    if candidate == old:
        return
    molecule = crystal._molecule
    coordinates = None
    if preserve == 'fractional' and molecule is not None:
        coordinates = _to_cartesian(_to_fractional(_coordinates(molecule), old[1]), candidate[1])
    _publish(crystal, candidate, transaction)
    if coordinates is not None:
        molecule.coordinates = coordinates


def _set_cell(crystal, matrix, preserve):
    with _scope(crystal) as transaction:
        _set_cell_locked(crystal, matrix, preserve, transaction)


def _set_parameter(crystal, index, value):
    with _scope(crystal) as transaction:
        matrix = _replace_parameter(crystal._state[0], index, value)
        _set_cell_locked(crystal, matrix, 'cartesian', transaction)


def _set_space_group(crystal, group, relations):
    with _scope(crystal) as transaction:
        old = _read_crystal_record(crystal)
        if relations is None:
            equivalent, occupancies, disorder = old[3]
            if equivalent and group is None:
                raise ValueError('CRYSTAL-IMAGE-MAP-INTEGRITY: explicit image relations require a known group')
            equivalent = tuple(tuple((row, SymmUnit.from_affine(operation.R, operation.t, context=group), image)
                                     for row, operation, image in entry) for entry in equivalent)
            relations = equivalent, occupancies, disorder
        candidate = _canonical_crystal_record((old[0], old[1], group, relations, *old[4:]),
                                               len(old[3][1]), len(old[4]))
        _publish(crystal, candidate, transaction)


def _set_site_relations(crystal, relations):
    with _scope(crystal) as transaction:
        old = _read_crystal_record(crystal)
        candidate = _canonical_crystal_record((*old[:3], relations, *old[4:]),
                                               len(old[3][1]), len(old[4]))
        _publish(crystal, candidate, transaction)


def _copy_crystal(crystal):
    from ..structure._crystal import _new_crystal
    molecule = crystal._molecule
    if molecule is None:
        return _new_crystal(_read_crystal_record(crystal), crystal_type=type(crystal))
    with molecule._mutation_lock:
        return molecule.copy().crystal


def _volume(matrix):
    from numpy.linalg import det
    return float(det(matrix))



def _full_image(key, group):
    row, operation, image = key
    return row, SymmUnit.from_affine(operation.R,
        tuple(value + shift for value, shift in zip(operation.t, image)), context=group)


def _image_entry(row, operation):
    representative, image = operation.reduce_to_cell()
    return row, representative, image


def _action_relations(relations, operation):
    equivalents, occupancies, disorder = relations
    inverse = ~operation
    transformed = tuple(tuple(_image_entry(row, operation @ member @ inverse)
        for row, member in (_full_image(key, operation.context) for key in entry))
        for entry in equivalents)
    return transformed, occupancies, disorder


def _origin_action(origin, operation):
    from ..rules.mathematics import symmetry as affine
    frame = origin[4]
    if operation.R == affine._IDENTITY and operation.t == affine._ZERO:
        return origin[1].identity()
    if frame is None:
        raise ValueError('CRYSTAL-UNSUPPORTED-PROJECTION: exact active-to-source basis map is unavailable')
    rotation = affine._mm(affine._mm(frame, operation.R), affine._inverse(frame))
    translation = affine._mv(frame, operation.t)
    return SymmUnit.from_affine(rotation, translation, context=origin[1])


def _action_origins(values, action):
    return tuple(None if entry is None else tuple(_image_entry(row, action @ member)
        for row, member in (_full_image(key, action.context) for key in entry))
        for entry in values)


def _axis_action(values, presence, sources, orthogonal):
    """Project actual frozen Cartesian subspaces into the existing axis flags."""
    import numpy as np
    from ..structure._fields import CONSTRAINT_FIELDS
    columns = tuple(FIELD_INDEX[name] for name in CONSTRAINT_FIELDS)
    bits = tuple(FIELD_PRESENCE_BITS[column] for column in columns)
    magnitude = np.abs(orthogonal)
    permutation = magnitude.argmax(axis=1)
    signed_permutation = (len(set(map(int, permutation))) == 3 and np.allclose(
        magnitude, np.eye(3)[permutation], rtol=0., atol=1e-10))
    previous_values = values[:, columns].copy()
    previous_sources = sources[:, columns].copy()
    supplied = np.column_stack(tuple((presence & bit) != 0 for bit in bits))
    if signed_permutation:
        values[:, columns] = previous_values[:, permutation]
        sources[:, columns] = previous_sources[:, permutation]
        presence &= np.uint32(~sum(bits) & ((1 << values.shape[1]) - 1))
        for target, bit in enumerate(bits):
            presence |= np.where(supplied[:, permutation[target]], bit, 0).astype('<u4')
        return
    for row in range(len(values)):
        if not supplied[row].any():
            continue
        if not supplied[row].all():
            raise ValueError('CRYSTAL-UNSUPPORTED-PROJECTION: incomplete Cartesian axis constraints have no rotated projection')
        projector = (orthogonal * previous_values[row]) @ orthogonal.T
        diagonal = np.diag(projector)
        flags = np.rint(diagonal)
        if not np.allclose(projector, np.diag(flags), rtol=0., atol=1e-10):
            raise ValueError('CRYSTAL-UNSUPPORTED-PROJECTION: rotated partial constraint is not Cartesian-axis aligned')
        values[row, columns] = flags
        if not np.array_equal(previous_values[row], flags):
            sources[row, columns] = ATOM_SOURCE_CODES['geometry']


def _operate_crystal(crystal, operation):
    import numpy as np
    from dataclasses import replace
    from ..rules.mathematics import symmetry as affine
    from ..rules.chemistry.stereochemistry.algebra import _transform_stereo_isometry
    from ..structure._fields import ATOM_FIELDS
    from ..structure._results import _copy_mca_result
    from ..structure.conformer import _capture_frame_records
    from ..structure.molecule import Molecule
    molecule = crystal._molecule
    if molecule is None:
        raise ValueError('CRYSTAL-UNSUPPORTED-PROJECTION: molecular action requires contained Molecule facts')
    with molecule._mutation_lock:
        if crystal._molecule is not molecule or molecule._crystal is not crystal:
            raise RuntimeError('CRYSTAL-CONTEXT-INTEGRITY: active association changed while acquiring the Molecule lock')
        if molecule._partial_charge_result is not None and not molecule._partial_charge_result.invalidated_by:
            from ..structure._identity import _check_source
            _check_source(molecule, molecule._partial_charge_result.source)
        record, metadata = molecule._capture_factual_record(include_frames=False)
        cell = record.cell
        group = cell[2]
        if group is None or not isinstance(operation, SymmUnit) or operation.context != group:
            raise ValueError('CRYSTAL-CONTEXT-INTEGRITY: operation requires the active known group')
        operation = SymmUnit.from_affine(operation.R, operation.t, context=group)
        coordinates = _coordinates(molecule)
        fractional = _to_fractional(coordinates, cell[1])
        transformed = _to_cartesian(operation.operate(fractional), cell[1])
        values = np.frombuffer(record.atom_values, dtype='<f8').reshape(-1, len(ATOM_FIELDS)).copy()
        presence = np.frombuffer(record.atom_presence, dtype='<u4').copy()
        sources = np.frombuffer(record.atom_sources, dtype='u1').reshape(values.shape).copy()
        coordinate_columns = tuple(FIELD_INDEX[name] for name in ('x', 'y', 'z'))
        changed = values[:, coordinate_columns] != transformed
        values[:, coordinate_columns] = transformed
        for offset, column in enumerate(coordinate_columns):
            sources[changed[:, offset], column] = ATOM_SOURCE_CODES['geometry']
        basis = np.asarray(cell[1]).T
        orthogonal = basis @ np.asarray(operation.R) @ np.linalg.inv(basis)
        _axis_action(values, presence, sources, orthogonal)
        stereo = _transform_stereo_isometry(record.stereo_payload, affine._det(operation.R))
        images = tuple(None if image is None else tuple(map(int, affine._mv(operation.R, image)))
                       for image in cell[4])
        origin, atom_origins, bond_origins = cell[6:]
        if origin is not None and any(entry is not None for entry in (*atom_origins, *bond_origins)):
            action = _origin_action(origin, operation)
            atom_origins = _action_origins(atom_origins, action)
            bond_origins = _action_origins(bond_origins, action)
        destination_cell = (*cell[:3], _action_relations(cell[3], operation), images,
                            cell[5], origin, atom_origins, bond_origins)
        frames, _generated, selection = _capture_frame_records(molecule, include_generated=False)
        prepared = replace(record, atom_values=values.tobytes(), atom_presence=presence.tobytes(),
                           atom_sources=sources.tobytes(), cell=destination_cell, stereo_payload=stereo,
                           source_frames=frames, conformer_results=(), frame_selection=selection)
        result = Molecule._build_molecule(prepared, bond_metadata=metadata)
        result.properties = dict(molecule.properties)
        _copy_mca_result(molecule, result)
        _invalidate_results(result, ('geometry', 'representation') if stereo != record.stereo_payload else ('geometry',))
        return result.crystal
