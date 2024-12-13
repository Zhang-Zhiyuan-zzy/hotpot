"""
python v3.9.0
@Project: hotpot
@File   : _core
@Auther : Zhiyuan Zhang
@Data   : 2024/12/5
@Time   : 10:59
"""
from abc import ABC, abstractmethod
from copy import copy
import numpy as np

class Conformers:
    def __init__(self, mol: "Molecule"):
        self.molecule = mol
        self.mol_conformer_idx = None

        self.coords = None
        self.charges = None
        self.energies = None

        self.idx = None


class Molecule:
    def __init__(self):
        self._atoms_data = None
        self._bonds_data = None
        self.atom_labels: list = []
        self.charges = 0

        self._atoms = []
        self._bonds = []
        self._conformers = Conformers(self)
        self.energy = 0.
        self.charge = 0

        self.zero_point = None
        self.free_energy = None  # - mol.energy - mol.zero_point  # Hartree to eV
        self.entropy = None
        self.enthalpy = None  # - mol.energy - mol.zero_point
        self.temperature = None
        self.pressure = None
        self.thermal_energy = None  # kcal to ev
        self.capacity = None  # cal to ev

    def _add_bond(self, bond: "Bond"):
        bond_attrs = np.zeros((1, len(bond.attrs_enumerator)))
        if self._bonds_data is None:
            self._bonds_data = bond_attrs
        else:
            self._bonds_data = np.vstack((self._bonds_data, bond_attrs))

        self._bonds.append(bond)

    def add_atom(self, atom: "Atom"):
        """ Create a new atom instance in the molecule. """
        if not isinstance(atom, Atom):
            raise TypeError("atom must be an instance of Atom")

        atom_attrs = np.zeros((1, len(atom.attrs_enumerator)))
        if self._atoms_data is None:
            self._atoms_data = atom_attrs
        else:
            self._atoms_data = np.vstack((self._atoms_data, atom_attrs))

        self._atoms.append(atom)
        atom.molecule = self

    def add_bond(self, atom1: "Atom", atom2: "Atom", bond_order=1., **kwargs):
        """ Create a new bond instance in the molecule. """
        if atom1.molecule is atom2.molecule is self:
            return Bond(atom1, atom2, bond_order, **kwargs)
        else:
            raise ValueError("at least one of atom1 and atom2 not on the molecule!")

    @property
    def atoms(self) -> list['Atom']:
        """ Return the interface to operate _atom_data """
        return copy(self._atoms)

    @property
    def bonds(self) -> list['Bond']:
        return copy(self._bonds)


class MolBlock(ABC):
    def __getattr__(self, item):
        try:
            attr_idx = self._attrs_enumerator.index(item)
            return self._attrs_dict[item](self.attrs[attr_idx])
        except ValueError:
            raise AttributeError(f"{item} is not an attribute of the {self.__class__.__name__}")

    def __setattr__(self, key, value):
        try:
            attr_idx = self._attrs_enumerator.index(key)
            self.attrs[attr_idx] = value
        except ValueError:
            super().__setattr__(key, value)

    def __repr__(self):
        return f"{self.__class__.__name__}({self.label})"

    @property
    def attrs_enumerator(self):
        return copy(self._attrs_enumerator)

    @property
    def label(self):
        raise NotImplementedError("Not implemented")


class MetaAtom(type):
    """ Metaclass to define Atom. """
    @staticmethod
    def _add_molecule_check(attr_name, attr_value):
        """ Add a check of whether the parent molecule exists for all properties of the Atom instance """
        def new_fget(self):
            if self.molecule is None:
                raise AttributeError(f"'molecule' is None when accessing property '{attr_name}'")
            return attr_value.fget(self)

        return property(
            fget=new_fget,
            fset=attr_value.fset,
            fdel=attr_value.fdel,
            doc=attr_value.__doc__
        )

    def __new__(mcs, name, bases, namespace):
        for attr_name, attr_value in namespace.items():
            if not attr_name.startswith("_") and isinstance(attr_value, property):
                namespace[attr_name] = MetaAtom._add_molecule_check(attr_name, attr_value)

        return super().__new__(mcs, name, bases, namespace)


class Atom(object, metaclass=MetaAtom):
    """ Represents an atom, which is an interface to operate _atom_data in its parent molecule. """
    _symbols = (
        "0",
        "H", "He", "Li", "Be", "B", "C", "N", "O", "F", "Ne",
        "Na", "Mg", "Al", "Si", "P", "S", "Cl", "Ar", "K", "Ca",
        "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn",
        "Ga", "Ge", "As", "Se", "Br", "Kr", "Rb", "Sr", "Y", "Zr",
        "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd", "In", "Sn",
        "Sb", "Te", "I", "Xe", "Cs", "Ba", "La", "Ce", "Pr", "Nd",
        "Pm", "Sm", "Eu", "Gd", "Tb", "Dy", "Ho", "Er", "Tm", "Yb",
        "Lu", "Hf", "Ta", "W", "Re", "Os", "Ir", "Pt", "Au", "Hg",
        "Tl", "Pb", "Bi", "Po", "At", "Rn", "Fr", "Ra", "Ac", "Th",
        "Pa", "U", "Np", "Pu", "Am", "Cm", "Bk", "Cf", "Es", "Fm",
        "Md", "No", "Lr", "Rf", "Db", "Sg", "Bh", "Hs", "Mt", "Ds",
        "Rg", "Cn", "Nh", "Fl", "Mc", "Lv", "Ts", "Og", "", ""
    )

    _attrs_dict = {
        # Name: datatype
        'atomic_number': int,
        'formal_charge': int,
        'partial_charge': float,
        'is_aromatic': bool,
        'x': float,
        'y': float,
        'z': float,
        'valence': int,
        'implicit_hydrogens': int,
        # 'explicit_hydrogens'
    }

    _attrs_enumerator = tuple(_attrs_dict.keys())

    def __init__(
            self,
            mol: Molecule = None,
            *,
            atomic_number: int = 0,
            atomic_symbol: str = 0,
            is_aromatic: int = 1,
            formal_charge: int = 0,
            partial_charge: int = 0.,
            coordinates: tuple[float, float, float] = (0., 0., 0.)
    ):
        kwargs = copy(locals()) # capturing all input arguments

        if isinstance(mol, Molecule):
            self.molecule = mol
        else:
            self.molecule = Molecule()

        atom_attrs = np.zeros((1, len(self._attrs_enumerator)))
        _atoms_data = getattr(self.molecule, "_atoms_data")
        if _atoms_data is None:
            _atoms_data = atom_attrs
        else:
            _atoms_data = np.vstack((_atoms_data, atom_attrs))

        setattr(self.molecule, "_atoms_data", _atoms_data)
        getattr(self.molecule, "_atoms").append(self)

        pop_items = ['self', 'mol']
        for item in pop_items:
            kwargs.pop(item)

        self.setattr(**kwargs)

    def __getattr__(self, item):
        try:
            attr_idx = self._attrs_enumerator.index(item)
            return self._attrs_dict[item](self.attrs[attr_idx])
        except ValueError:
            raise AttributeError(f"{item} is not an attribute of the {self.__class__.__name__}")

    def __setattr__(self, key, value):
        try:
            attr_idx = self._attrs_enumerator.index(key)
            self.attrs[attr_idx] = value
        except ValueError:
            super().__setattr__(key, value)

    def __dir__(self):
        return ['idx', 'atomic_number', 'atomic_symbol', 'coordinates']

    def __repr__(self):
        return f"{self.__class__.__name__}({self.label})"

    @property
    def idx(self):
        """ The atom index in the Molecule """
        return self.molecule.atoms.index(self)

    @property
    def attrs_enumerator(self):
        return self._attrs_enumerator

    @property
    def attrs(self) -> np.ndarray:
        return getattr(self.molecule, "_atoms_data")[self.idx]

    @property
    def coordinates(self):
        return self.x, self.y, self.z

    @property
    def symbol(self):
        return self._symbols[self.atomic_number]

    @symbol.setter
    def symbol(self, value: str):
        self.atomic_number = self._symbols.index(value)

    @property
    def label(self) -> str:
        return f"{self.symbol}{self.idx}"

    def setattr(self, **kwargs):
        for name, value in kwargs.items():
            if name == 'coordinates':
                setattr(self, 'x', value[0])
                setattr(self, 'y', value[1])
                setattr(self, 'z', value[2])
            else:
                setattr(self, name, value)


class Bond(MolBlock):

    _attrs_enumerator = ('bond_order',)
    _bond_order_symbol = {
        1.: '-',
        1.5: '@',
        2.: '=',
        3.: '#'
    }
    _bond_order_names = {
        1.: 'Single',
        1.5: 'Aromatic',
        2.: 'Double',
        3.: 'Triple'
    }

    def __init__(self, atom1: Atom, atom2: Atom, bond_order: float = 1., **kwargs):
        if atom1.molecule is not atom2.molecule:
            raise ValueError("The two atoms must on same Molecule object")

        self._atom1 = atom1
        self._atom2 = atom2

        getattr(self.molecule, "_add_bond")(self)

        kwargs['bond_order'] = bond_order
        self.set_attrs(**kwargs)

    def set_attrs(self, **kwargs) -> None:
        for name, value in kwargs.items():
            setattr(self, name, value)

    @property
    def atom1(self) -> Atom:
        return self._atom1

    @property
    def atom2(self) -> Atom:
        return self._atom2

    @property
    def molecule(self) -> Molecule:
        return self._atom1.molecule

    @property
    def attrs(self) -> np.ndarray:
        return getattr(self.molecule, "_bonds_data")[self.idx]

    @property
    def idx(self) -> int:
        return self.molecule.bonds.index(self)

    @property
    def label(self):
        return f"{self._atom1.label}{self._bond_order_symbol[self.bond_order]}{self._atom2.label}"


if __name__ == "__main__":
    a1, a2 = Atom(), Atom()
    # a.molecule = Molecule()
    print(a1.idx)
    a1.symbol = "Am"
    a2.symbol = "Eu"
    print(a1.atomic_number)
    print(a2.atomic_number)

    print(a1.molecule is a2.molecule)
    a1.molecule.add_atom(a2)
    print(a1.molecule is a2.molecule)
    b = a1.molecule.add_bond(a1, a2, 2)

    print(b.idx)
    print(b.label)
