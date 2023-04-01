# Molecular Descriptor Toolkit

This code provides a toolkit for working with molecular descriptors using the Open Babel library. The classes provided allow for the creation and manipulation of molecules, atoms, bonds, and crystals. 

## Molecule

The `Molecule` class represents a chemical molecule and contains information about the atoms and bonds in the molecule. The class has the following methods:

- `__init__(self, file=None, data=None, fmt=None, **kwargs)`: Initialize a new `Molecule` instance.
- `__repr__(self)`: Return a string representation of the `Molecule` instance.
- `add_atom(self, atomic_num: int, coords: Union[Tuple[float, float, float], np.ndarray], **kwargs)`: Add a new atom to the molecule.
- `add_bond(self, atom1: Atom, atom2: Atom, bond_order: int)`: Add a new bond between two atoms in the molecule.
- `copy(self)`: Create a copy of the `Molecule` instance.
- `dump(self, fmt: str, **kwargs)`: Generate a string representation of the molecule in a specific file format.
- `element_feature(self, *feature_name: str)`: Get a vector of the specified element feature names for each atom in the molecule.
- `from_file(cls, file: Union[str, PathLike], fmt: Optional[str] = None)`: Create a new `Molecule` instance from a file.
- `from_string(cls, data: str, fmt: Optional[str] = None)`: Create a new `Molecule` instance from a string.
- `get_bond(self, atom1: Atom, atom2: Atom)`: Get the bond between two atoms in the molecule.
- `get_ring(self, size: int)`: Get all rings of a specific size in the molecule.
- `get_smiles(self, isomeric: bool = True)`: Get the SMILES representation of the molecule.
- `remove_bond(self, bond: Bond)`: Remove a bond from the molecule.
- `remove_atom(self, atom: Atom)`: Remove an atom and all bonds connected to it from the molecule.
- `set_coords(self, coords: np.ndarray)`: Set the coordinates of all atoms in the molecule.
- `to_deepmd_train_data(self, path_save: Union[str, PathLike], valid_set_size: Union[int, float] = 0.2, is_test_set: bool = False, _call_by_bundle: bool = False)`: Convert the `Molecule` instance to a DeepMD training data file.
- `writefile(self, fmt: str, path_file, *args, **kwargs)`: Write the molecule to a file in a specific format.

## Atom

The `Atom` class represents an individual atom in a molecule and contains information about its atomic number, coordinates, and bonds to other atoms. The class has the following methods:

- `__init__(self, OBAtom: ob.OBAtom = None, **kwargs)`: Initialize a new `Atom` instance.
- `__repr__(self)`: Return a string representation of the `Atom` instance.
- `copy(self)`: Create a copy of the `Atom` instance.
- `element_feature(self, *feature_name: str)`: Get a vector of the specified element feature names for the atom.
- `neighbours(self)`: Get all atoms bonded to the atom in the same molecule.

## Bond

The `Bond` class represents a chemical bond between two atoms in a molecule. The class has the following methods:

- `__init__(self, _OBBond: ob.O
