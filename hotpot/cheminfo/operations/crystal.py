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
    candidate = _canonical_crystal_record((old[0], matrix, *old[2:]), len(old[3][1]), len(old[4]))
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
