"""Fixed atom-sequence membership over factual carriers."""

import re
from typing import Union

from ..structure.atom import Atom
from ..structure.molecule import Molecule
from ..structure._commit import _commit_topology
from ..structure._identity import _capture_source


class AtomSeq:
    """
    Represent instances assembled by a sequence of atoms, like Bond, Angles, Torison, and Rings.
    This class represents a sequence of atoms and provides various utilities for working
    with atoms, their indices, bonds, and molecular structures related to the sequence.

    The AtomSeq class is designed to manage a sequence of atoms that belong to the same
    molecule. It validates the consistency of provided atoms, such as ensuring all atoms
    belong to the same molecule and verifying the expected number of atoms. The class provides
    methods and attributes for accessing atoms, their indices, associated bonds, and other
    structural information. It also supports conversion of the represented sequence to
    a new Molecule instance.

    Attributes:
        _length (int or None): Expected fixed length for atoms in sub-classes or None if no constraint exists.
        _atoms (tuple[Atom]): A tuple containing the sequence of Atom objects.
        _bonds (list[Bond]): A list containing Bond objects derived from the atom sequence.
    """
    _length = None

    def __setattr__(self, name, value):
        if '_view_source' in self.__dict__ and (
                name in ('_atoms', '_bonds') or name.startswith('_view_') or re.fullmatch(r'(?:atom|bond)\d+|a\d+idx', name)):
            raise AttributeError('View membership is read-only')
        super().__setattr__(name, value)

    def __reduce_ex__(self, protocol):
        if '_view_source' not in self.__dict__:
            return object.__reduce_ex__(self, protocol)
        from ._source import _restore_sequence_view, _view_source_current
        with self.mol._mutation_lock:
            current = _view_source_current(self.mol, self._view_source, self._atoms, self._view_uids)
            if '_view_representation' in self.__dict__:
                from ._source import _restore_ring_view
                return (_restore_ring_view, (type(self), self._atoms, self._bonds,
                         self._view_geometry, self._view_representation, current))
            return (_restore_sequence_view,
                    (type(self), self._atoms, self._bonds, self._view_geometry, current))

    def __init__(self, *atoms: Atom):
        from ..structure.bond import Bond
        from .topology import Ring
        self._check_is_same_mol(*atoms)
        self._check_atom_number(*atoms)
        self._atoms = atoms

        with self.mol._mutation_lock:
            if isinstance(self, Bond):
                self._bonds = [self]
            else:
                self._bonds = [self.mol.bond(atoms[i].idx, atoms[i+1].idx) for i in range(len(atoms) - 1)]
            if not isinstance(self, (Bond, Ring)):
                from ._source import _bind_view
                self._bonds = tuple(self._bonds)
                _bind_view(self, atoms)

    def __repr__(self):
        return f"{self.__class__.__name__}" + '(' + ''.join(a.symbol for a in self.atoms) + ')'

    def __getattr__(self, item):
        if re.match(r"atom\d+", item):
            idx = int(item[4:]) - 1
            with self.mol._mutation_lock:
                return self._atoms[idx]

        elif re.match(r"bond\d+", item):
            idx = int(item[4:]) - 1
            if '_view_source' in self.__dict__:
                return self._bonds[idx]
            atom_start = self._atoms[idx]
            atom_end = self._atoms[idx+1]
            return list(set(atom_start.bonds) & set(atom_end.bonds))[0]

        elif re.match(r"a\d+idx", item):
            len_idx = len(item) - 4
            idx = int(item[1:1+len_idx]) - 1
            with self.mol._mutation_lock:
                self._check_topology_source()
                return self._atoms[idx].idx

        else:
            super().__getattribute__(item)

    def __len__(self):
        return len(self._atoms)

    def __contains__(self, item: Union[Atom, "Bond"]):
        from ..structure.bond import Bond
        if isinstance(item, Atom):
            return item in self._atoms
        elif isinstance(item, Bond):
            return item in self._bonds

    def __eq__(self, other):
        if not isinstance(other, self.__class__):
            return False

        if len(self.atoms) != len(other.atoms) or len(self.bonds) != len(other.bonds):
            return False

        return all(a in self._atoms for a in other.atoms) and all(b in self._bonds for b in other.bonds)

    def _replace_atom(self, old: Union[int, Atom], new: Atom):
        if '_view_source' in self.__dict__:
            raise AttributeError('View membership is read-only')
        assert isinstance(new, Atom) and new not in self
        if isinstance(old, int):
            if 0 <= old < len(self._atoms):
                raise ValueError(f'The atom index in Bond must be greater than or equal to 0 and less than {len(self)}')
            self._atoms = tuple(a if i != old else new for i, a in enumerate(self._atoms))
        elif isinstance(old, Atom):
            assert old in self
            self._atoms = tuple(a if a is not old else new for a in self._atoms)
        else:
            raise TypeError(f"The old atom should be given by int or Atom, instead of a {type(old)}")

    def _check_topology_source(self):
        if '_view_source' in self.__dict__:
            from ._source import _check_view_source
            _check_view_source(self.mol, self._view_source, self._atoms, self._view_uids)

    @staticmethod
    def _check_is_same_mol(*atoms):
        if any(atoms[0].mol is not a.mol for a in atoms[1:]):
            raise ValueError('All atoms must belong to same mol.')

    def _check_atom_number(self, *atoms):
        _length = getattr(self, '_length', None)
        if _length and len(atoms) != _length:
            raise ValueError(
                f"The the atom counts of {self.__class__.__name__} is {_length}, but {len(self.atoms)} are given.")

    @property
    def atoms(self) -> tuple[Atom, ...]:
        with self.mol._mutation_lock:
            return tuple(self._atoms)

    @property
    def atoms_indices(self):
        with self.mol._mutation_lock:
            self._check_topology_source()
            return [a.idx for a in self._atoms]

    @property
    def bonds(self) -> list["Bond"]:
        return list(self._bonds)

    @property
    def mol(self) -> Molecule:
        return self._atoms[0].mol

    @property
    def has_metal(self):
        with self.mol._mutation_lock:
            self._check_topology_source()
            return any(a.is_metal for a in self._atoms)

    def to_mol(self):
        with self.mol._mutation_lock:
            self._check_topology_source()
            record, metadata = self.mol._capture_factual_record(rows=tuple(atom.idx for atom in self._atoms), include_frames=False)
        mol = Molecule()
        _commit_topology(mol, _capture_source(mol), record,
                         atom_bindings=(None,) * record.atom_count,
                         bond_bindings=(None,) * len(record.bond_endpoints), bond_metadata=metadata)
        return mol



def _new_sequence_view(cls, atoms, rows, complete_paths, *, source, uids, geometry):
    from ._source import _initialize_view_source, _member_geometry
    view = object.__new__(cls)
    members = tuple(atoms[row] for row in rows)
    object.__setattr__(view, '_atoms', members)
    object.__setattr__(view, '_bonds', tuple(complete_paths.get(frozenset((id(first), id(second))))
                                          for first, second in zip(members, members[1:])))
    _initialize_view_source(view, source, tuple(uids[row] for row in rows),
                            _member_geometry(geometry, rows, len(atoms)))
    return view
