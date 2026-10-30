"""Task-local view source capabilities over the one molecular topology index."""

from ..structure._fields import FIELD_INDEX, FIELD_PRESENCE_BITS
from ..structure._identity import _capture_source, _check_source


__all__ = ()

_COORDINATES = ("x", "y", "z")
_INDICES = tuple(FIELD_INDEX[name] for name in _COORDINATES)
_PRESENCE = sum(FIELD_PRESENCE_BITS[index] for index in _INDICES)


def _capture_view_source(mol):
    return (_capture_source(mol), mol._topology_index, len(mol._atoms),
            len(mol._bonds), len(mol._hided_metal_bonds), len(mol._hided_covalent_bonds))


def _check_view_source(mol, source, atoms, uids):
    _check_source(mol, source[0])
    if not _view_source_current(mol, source, atoms, uids):
        raise RuntimeError("View topology source changed; request a new view")


def _view_source_current(mol, source, atoms, uids):
    return (source[0].owner is mol._owner_token and source[1] is mol._topology_index
            and source[2:] == (len(mol._atoms), len(mol._bonds),
                               len(mol._hided_metal_bonds), len(mol._hided_covalent_bonds))
            and all(atom._uid_owner is mol._owner_token and atom._uid == uid
                    and id(atom) in mol._topology_index for atom, uid in zip(atoms, uids)))


def _geometry_snapshot(mol, atoms):
    rows = tuple(mol._topology_index.get(id(atom), (None,))[0] for atom in atoms)
    if any(row is None for row in rows):
        return None, None
    table = mol._atom_table
    coordinates = table.select(_COORDINATES, rows=rows)
    presence = (table._presence[list(rows)] & _PRESENCE).tobytes()
    sources = table._sources[list(rows)][:, _INDICES].tobytes()
    return coordinates, (coordinates.tobytes(), presence, sources)


def _member_geometry(geometry, rows, count):
    return tuple(b''.join(
        data[row * (len(data) // count):(row + 1) * (len(data) // count)]
        for row in rows) for data in geometry)


def _view_coordinates(view):
    mol = view.mol
    with mol._mutation_lock:
        _check_view_source(mol, view._view_source, view.atoms, view._view_uids)
        coordinates, identity = _geometry_snapshot(mol, view.atoms)
        if identity != view._view_geometry:
            raise RuntimeError("View geometry source changed; request a new view")
        return coordinates


def _initialize_view_source(view, source, uids, geometry):
    """Install the existing source fields on a newly constructed view."""
    object.__setattr__(view, '_view_source', source)
    object.__setattr__(view, '_view_uids', uids)
    object.__setattr__(view, '_view_geometry', geometry)


def _bind_view(view, atoms, *, source=None):
    mol = atoms[0].mol
    with mol._mutation_lock:
        _initialize_view_source(
            view, _capture_view_source(mol) if source is None else source,
            tuple(atom._uid for atom in atoms), _geometry_snapshot(mol, atoms)[1])


def _ring_representation(mol, atoms, bonds):
    table = mol._atom_table
    return (tuple((int(table._columns[0][mol._topology_index[id(atom)][0]]),
                   bool(table._columns[FIELD_INDEX['is_aromatic']][mol._topology_index[id(atom)][0]]))
                  for atom in atoms),
            tuple(None if bond is None else (float(bond._attrs[0]), bond._bond_kind.value, bond._is_aromatic)
                  for bond in bonds))


def _check_ring_source(view, *, prepare=False):
    mol = view.mol
    with mol._mutation_lock:
        _check_view_source(mol, view._view_source, view.atoms, view._view_uids)
        if prepare:
            from ..operations.chemistry import _prepare_aromatic_representation
            _prepare_aromatic_representation(mol, rows=tuple(mol._topology_index[id(atom)][0] for atom in view.atoms))
        if _ring_representation(mol, view.atoms, view.bonds) != view._view_representation:
            raise RuntimeError('Ring representation source changed; request a new view')


def _restored_source(mol, current):
    source = _capture_view_source(mol)
    return source if current else (source[0], object(), *source[2:])


def _restore_sequence_view(cls, atoms, bonds, geometry, current):
    view = object.__new__(cls)
    view._atoms, view._bonds = atoms, bonds
    _bind_view(view, atoms, source=_restored_source(atoms[0].mol, current))
    object.__setattr__(view, '_view_geometry', geometry)
    return view


def _restore_ring_view(cls, atoms, bonds, geometry, representation, current):
    view = _restore_sequence_view(cls, atoms, bonds, geometry, current)
    object.__setattr__(view, '_view_representation', representation)
    return view


def _restore_joint_ring(cls, rings, atoms, bonds, geometry, representation, current):
    view = object.__new__(cls)
    for name, value in (('_member_rings', rings), ('_atoms', atoms), ('_bonds', bonds), ('_mol', atoms[0].mol)):
        object.__setattr__(view, name, value)
    _bind_view(view, atoms, source=_restored_source(atoms[0].mol, current))
    object.__setattr__(view, '_view_geometry', geometry)
    object.__setattr__(view, '_view_representation', representation)
    return view


def _restore_pair_view(cls, atoms, geometry, current):
    pair = object.__new__(cls)
    pair._atoms = atoms
    _bind_view(pair, atoms, source=_restored_source(atoms[0].mol, current))
    pair._view_geometry = geometry
    return pair


def _restore_pair_table(cls, mol, atoms, geometry, current, items):
    from types import MappingProxyType
    table = object.__new__(cls)
    table._mol, table._atoms = mol, atoms
    table._members = MappingProxyType({atom: row for row, atom in enumerate(atoms)})
    table._items = None if items is None else MappingProxyType(dict(items))
    table._view_source = _restored_source(mol, current)
    table._view_uids = tuple(atom._uid for atom in atoms)
    table._geometry = geometry
    return table
