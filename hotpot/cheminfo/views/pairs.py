"""Fixed atom-pair views and caller-owned pair tables."""

from itertools import combinations
from typing import Mapping

import numpy as np

from ..structure.atom import Atom


class AtomPair:
    """Fixed pair membership with source-checked mathematics and supplied observations."""

    __slots__ = ('_atoms', '_view_source', '_view_uids', '_view_geometry')
    _length = 2
    attr_names = ('wiberg_bond_order', 'length_shortest_path')

    def __init__(self, atom1: Atom, atom2: Atom):
        from ._source import _bind_view
        assert atom1.mol is atom2.mol
        self._atoms = (atom1, atom2)
        _bind_view(self, self._atoms)

    def __reduce_ex__(self, protocol):
        from ._source import _restore_pair_view, _view_source_current
        with self.mol._mutation_lock:
            current = _view_source_current(self.mol, self._view_source, self._atoms, self._view_uids)
            return _restore_pair_view, (type(self), self._atoms, self._view_geometry, current)

    @property
    def atoms(self):
        return self._atoms

    @property
    def atom1(self):
        return self._atoms[0]

    @property
    def atom2(self):
        return self._atoms[1]

    @property
    def mol(self):
        return self.atom1.mol

    def __repr__(self):
        from ._source import _check_view_source
        with self.mol._mutation_lock:
            _check_view_source(self.mol, self._view_source, self._atoms, self._view_uids)
            return f"AtomPair({self.atom1.idx}, {self.atom2.idx})"

    def __eq__(self, other):
        if not isinstance(other, AtomPair):
            return NotImplemented
        return frozenset(self._atoms) == frozenset(other._atoms)

    def __contains__(self, atom):
        return atom is self.atom1 or atom is self.atom2

    @property
    def wiberg_bond_order(self):
        with self.mol._mutation_lock:
            if any(atom._uid_owner is not self.mol._owner_token or atom._uid != uid
                   or id(atom) not in self.mol._topology_index
                   for atom, uid in zip(self._atoms, self._view_uids)):
                raise RuntimeError('Pair observation members changed')
            return self.mol._pair_wiberg_bond_orders.get(tuple(sorted(self._view_uids)), 0)

    @wiberg_bond_order.setter
    def wiberg_bond_order(self, value):
        from ..structure._commit import _commit_pair_wiberg
        _commit_pair_wiberg(self.mol, self._view_source[0], self._atoms, self._view_uids, value)

    @property
    def attrs(self):
        return [getattr(self, name) for name in self.attr_names]

    @property
    def shortest_path(self) -> list['Atom']:
        from ._source import _check_view_source
        with self.mol._mutation_lock:
            _check_view_source(self.mol, self._view_source, self._atoms, self._view_uids)
            return self.mol.shortest_path(self.atom1, self.atom2)

    @property
    def length_shortest_path(self) -> int:
        return len(self.shortest_path)

    @property
    def distance(self):
        from ..operations.geometry import _atom_pair_distance
        return _atom_pair_distance(self)



class AtomPairs(Mapping):
    """Caller-owned fixed pair table; scientific observations remain at their owner."""

    __slots__ = ('_mol', '_atoms', '_members', '_items', '_view_source', '_view_uids', '_geometry')

    def __init__(self, *args, mol, **kwargs):
        from types import MappingProxyType
        from ._source import _capture_view_source, _geometry_snapshot
        with mol._mutation_lock:
            self._mol = mol
            self._atoms = tuple(mol._atoms)
            self._members = MappingProxyType({atom: row for row, atom in enumerate(self._atoms)})
            self._items = MappingProxyType(dict(*args, **kwargs)) if args or kwargs else None
            self._view_source = _capture_view_source(mol)
            self._view_uids = tuple(atom._uid for atom in self._atoms)
            self._geometry = _geometry_snapshot(mol, self._atoms)[1]

    def __reduce_ex__(self, protocol):
        from ._source import _restore_pair_table, _view_source_current
        with self.mol._mutation_lock:
            current = _view_source_current(self.mol, self._view_source, self._atoms, self._view_uids)
            items = None if self._items is None else tuple(self._items.items())
            return (_restore_pair_table,
                    (type(self), self.mol, self._atoms, self._geometry, current, items))

    @property
    def mol(self):
        return self._mol

    def __len__(self):
        count = len(self._atoms)
        return len(self._items) if self._items is not None else count * (count - 1) // 2

    def __iter__(self):
        return iter(self._items) if self._items is not None else (
            frozenset(atoms) for atoms in combinations(self._atoms, 2))

    def __getitem__(self, atoms):
        key = frozenset(atoms)
        if self._items is not None:
            return self._items[key]
        if len(key) != 2 or any(atom not in self._members for atom in key):
            raise KeyError(key)
        atoms = tuple(atoms)
        pair = object.__new__(AtomPair)
        pair._atoms = atoms
        pair._view_source = self._view_source
        pair._view_uids = tuple(atom._uid for atom in atoms)
        rows = tuple(self._members[atom] for atom in atoms)
        from ._source import _member_geometry
        pair._view_geometry = _member_geometry(self._geometry, rows, len(self._atoms))
        return pair

    def getdefault(self, atom_pair):
        return self[atom_pair]

    def copy(self):
        return dict(self.items())

    def update_pairs(self):
        from ._source import _check_view_source
        with self.mol._mutation_lock:
            _check_view_source(self.mol, self._view_source, self._atoms, self._view_uids)

    @property
    def pair_distance(self):
        from ..operations.geometry import _atom_pairs_distance
        return _atom_pairs_distance(self)

    def pairs_shortest_path(self) -> dict:
        return {key: pair.length_shortest_path for key, pair in self.items()}

    @property
    def idx_matrix(self) -> np.ndarray:
        with self.mol._mutation_lock:
            self.update_pairs()
            return np.array([[atom.idx for atom in key] for key in self], dtype=np.int64)
