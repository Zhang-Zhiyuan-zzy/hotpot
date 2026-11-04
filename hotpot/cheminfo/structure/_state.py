"""Current-version molecular graph capture and restoration."""

from ._atom_table import _AtomFactTable
from .bond_kind import BondKind
from .constraints import _ConstraintSet, _bind_constraint_members
from ._dependencies import _dependency_state
from .stereo import StereoPayload as _StereoPayload
from ._stereo import _install_stereo_storage
from .conformer import _install_frame_records


def _capture_molecule_pickle(mol):
    with mol._mutation_lock:
        record, metadata = mol._capture_factual_record(copy_metadata=False)
        bonds = tuple(mol._bonds)
        owned = {'_mutation_lock', '_owner_token', '_atoms', '_bonds',
                 '_atom_table', '_topology_index', '_next_atom_uid', '_next_bond_uid', '_conformers',
                 '_conformers_index', '_stereo_schema', '_stereo_records', '_stereo_source',
                 '_crystal', '_charge_value', '_charge_present',
                 '_charge_source', '_spin_value', '_spin_present', '_spin_source',
                 '_constraints', '_pair_wiberg_bond_orders', '_source_frames',
                 '_conformer_results', '_frame_result'}
        state = {key: value for key, value in mol.__dict__.items() if key not in owned}
        state['properties'] = dict(mol.properties)
        state['_crystal'] = mol._crystal
        graph_memo = {id(mol): mol, **{id(atom): atom for atom in mol._atoms},
                      **{id(bond): bond for bond in bonds}}
        atom_state = tuple({key: value for key, value in atom.__dict__.items()
                            if key not in ('_mol', '_member_uid', '_uid_owner', '_detached_table', '_constraint_values', 'frac_valence')}
                           for atom in mol._atoms)
        bond_state = tuple({key: value for key, value in bond.__dict__.items()
                            if key not in ('_atoms', '_bonds', '_member_uid', '_uid_owner', '_attrs',
                                           '_bond_kind', '_bond_direction', '_display_facts', '_bond_source', '_bond_source_metadata', '_is_aromatic', '_constraint_values')}
                           for bond in bonds)
        extra = (state, metadata, atom_state, bond_state)
        return (record, mol._owner_token, tuple(atom._uid for atom in mol._atoms),
                tuple(bond._uid for bond in bonds), (mol._next_atom_uid, mol._next_bond_uid),
                _dependency_state(mol._atom_table), extra)


def _restore_molecule_raw(cls, record, owner, atom_uids, bond_uids, high_water, dependencies):
    mol = cls()
    mol._owner_token = owner
    mol._install_factual_record(record, dependency_state=dependencies, _install_frames=False, _install_crystal=False)
    bonds = tuple(mol._bonds)
    atom_cells = tuple(atom._constraint_values[atom._uid] for atom in mol._atoms)
    bond_cells = tuple(bond._constraint_values[bond._uid] for bond in bonds)
    for atom, uid in zip(mol._atoms, atom_uids):
        object.__setattr__(atom, '_member_uid', uid)
    for bond, uid in zip(bonds, bond_uids):
        object.__setattr__(bond, '_member_uid', uid)
    mol._constraints = _ConstraintSet(
        tuple(tuple(atom_uids[row] for row in entry) for entry in record.angle_constraints),
        tuple(tuple(atom_uids[row] for row in entry) for entry in record.torsion_constraints),
        dict(zip(atom_uids, atom_cells)), dict(zip(bond_uids, bond_cells)))
    _bind_constraint_members(mol)
    mol._pair_wiberg_bond_orders = {
        tuple(sorted((atom_uids[first], atom_uids[second]))): value
        for first, second, value in record.pair_wiberg_bond_orders}
    mol._next_atom_uid, mol._next_bond_uid = high_water
    _install_stereo_storage(mol, record.stereo_payload if record.stereo_payload is not None
                            else _StereoPayload(1, (), (), ()))
    from ._crystal import _install_crystal_record
    _install_crystal_record(mol, record.cell)
    _install_frame_records(mol, record.source_frames, record.conformer_results, record.frame_selection)
    return mol


def _restore_molecule_extra(mol, state):
    attributes, metadata, atom_state, bond_state = state
    mol.__dict__.update(attributes)
    for atom, extra in zip(mol._atoms, atom_state):
        atom.__dict__.update(extra)
    for bond, extra, supplied in zip(mol._bonds, bond_state, metadata):
        bond.__dict__.update(extra)
        object.__setattr__(bond, '_bond_source_metadata', supplied)


def _restore_atom_member(mol, row, uid):
    if row >= len(mol._atoms) or mol._atoms[row]._uid != uid:
        raise RuntimeError('Serialized Atom membership changed during capture')
    return mol._atoms[row]


def _restore_bond_member(mol, row, uid):
    bond = _bond_member_at(mol, row)
    if bond is None or bond._uid != uid:
        raise RuntimeError('Serialized Bond membership changed during capture')
    return bond


def _bond_member_at(mol, row):
    return mol._bonds[row] if row < len(mol._bonds) else None


def _restore_detached_atom(mol, uid, state, cell):
    from .atom import _new_atom_handle, _seal_detached_table
    atom = _new_atom_handle(mol, uid, cell=cell)
    table = object.__new__(_AtomFactTable)
    table.__setstate__(state)
    object.__setattr__(atom, '_detached_table', _seal_detached_table(table))
    return atom


def _restore_unregistered_bond(mol, uid, first, second, values):
    from .bond import _new_bond_handle
    return _new_bond_handle(mol, uid, first, second, values, kind=BondKind.UNKNOWN,
                            direction=None, source_namespace=None, metadata={})
