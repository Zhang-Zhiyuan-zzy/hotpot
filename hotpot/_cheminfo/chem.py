"""
python v3.9.0
@Project: hotpot
@File   : molecule
@Auther : Zhiyuan Zhang
@Data   : 2023/10/14
@Time   : 16:25
"""
from abc import ABC
import weakref

from openbabel import openbabel as ob, pybel as pb

from ._base import Wrapper


_molecule_dict = weakref.WeakValueDictionary()


def _refcode_getter(ob_mol: ob.OBMol) -> int:
    """ retrieve refcode from given OBMol object """
    return int(ob.toCommentData(ob_mol.GetData("refcode")).GetData())


class MolUnit(Wrapper, ABC):
    """ defining the base methods for Molecule Units ,like Atom and Bonds """
    @property
    def molecule(self) -> "Molecule":
        """ the parent molecule """
        return _molecule_dict[_refcode_getter(self._obj.GetParent())]


class Molecule(Wrapper, ABC):
    """Represent an intuitive molecule"""
    def __init__(self, ob_mol):
        super().__init__(ob_mol)
        self._set_refcode()

    def __repr__(self):
        return f"Mol({self.ob_mol.GetFormula()})"

    def _set_refcode(self):
        """ put an int value into the OBMol as the refcode """
        if not self.refcode:
            self._set_ob_int_data('refcode', 0 if not _molecule_dict else max(_molecule_dict.keys()) + 1)
            _molecule_dict[self.refcode] = self

    @property
    def atoms(self) -> list["Atom"]:
        return [Atom(oba) for oba in ob.OBMolAtomIter(self.ob_mol)]

    @property
    def bonds(self) -> list["Bond"]:
        return [Bond(obb) for obb in ob.OBMolBondIter(self.ob_mol)]

    @property
    def ob_mol(self) -> ob.OBMol:
        return self._obj

    @property
    def refcode(self) -> int:
        """ get the refcode of this molecule in the molecular WeakValueDictionary """
        return self._get_ob_int_data('refcode')


class Atom(MolUnit):
    """Represent an intuitive Atom"""
    def __repr__(self):
        return f"Atom({self.label})"

    @property
    def ob_atom(self) -> ob.OBAtom:
        return self._obj

    @property
    def label(self) -> str:
        return self._get_ob_comment_data('label') or self.symbol

    @property
    def symbol(self) -> str:
        return ob.GetSymbol(self.ob_atom.GetAtomicNum())


class Bond(MolUnit):
    """ Represent an intuitive Bond """
    _type_name = {
        0: 'Unknown',
        1: 'Single',
        2: 'Double',
        3: 'Triple',
        5: 'Aromatic'
    }

    @property
    def ob_bond(self) -> ob.OBBond:
        return self._obj

    def __repr__(self):
        return f"Bond({self.atom1.label}, {self.atom2.label}, {self.type_name})"

    @property
    def atom1(self) -> Atom:
        return Atom(self.ob_bond.GetBeginAtom())

    @property
    def atom2(self) -> Atom:
        return Atom(self.ob_bond.GetEndAtom())

    @property
    def type(self) -> int:
        return self.ob_bond.GetBondOrder()

    @type.setter
    def type(self, bond_order: int):
        self.ob_bond.SetBondOrder(bond_order)

    @property
    def type_name(self):
        return self._type_name[self.type]
