"""
python v3.9.0
@Project: hotpot
@File   : cryst
@Auther : Zhiyuan Zhang
@Data   : 2023/10/14
@Time   : 16:31
"""
import weakref
from abc import ABC
from typing import *

import numpy as np
from openbabel import openbabel as ob

from . import _chem
from ._base import Wrapper


_crystal_dict = weakref.WeakValueDictionary()


class Crystal(Wrapper, ABC):
    """"""
    _lattice_type = (
        'Undefined', 'Triclinic', 'Monoclinic', 'Orthorhombic', 'Tetragonal', 'Rhombohedral', 'Hexagonal', 'Cubic'
    )

    def __new__(cls, cell: ob.OBUnitCell, mol=None):
        if not mol:
            return super().__new__(cls)
        else:
            return _crystal_dict.setdefault(mol, super().__new__(cls))

    def __init__(self, cell: ob.OBUnitCell, mol=None):
        super().__init__(cell)
        self.molecule = mol

    def __repr__(self):
        return f'Crystal({self.lattice_type}, {self.space_group}, {self.molecule})'

    def __eq__(self, other):
        return bool(self.molecule) and self.molecule == other.molecule

    def __hash__(self):
        return hash(f"hotpot.Crystal(refcode={self.molecule.refcode})")

    @property
    def ob_unit_cell(self) -> ob.OBUnitCell:
        return self._obj

    @staticmethod
    def _matrix_to_params(matrix: np.ndarray):
        """ Covert the cell matrix to cell parameters: a, b, c, alpha, beta, gamma """
        va, vb, vc = matrix
        a = sum(va ** 2) ** 0.5
        b = sum(vb ** 2) ** 0.5
        c = sum(vc ** 2) ** 0.5

        alpha = np.arccos(np.dot(va, vb) / (a * b)) / np.pi * 180
        beta = np.arccos(np.dot(va, vc) / (a * c)) / np.pi * 180
        gamma = np.arccos(np.dot(vb, vc) / (b * c)) / np.pi * 180

        return a, b, c, alpha, beta, gamma

    def _set_space_group(self, space_group: str):
        self.ob_unit_cell.SetSpaceGroup(space_group)

    @property
    def density(self) -> float:                         # Avogadro / angstrom^3
        """ The density with kg/m^3 """
        return self.pack_molecule.weight / self.volume * (6.02214076e23*1e-30)

    @property
    def lattice_type(self) -> str:
        return self._lattice_type[self.ob_unit_cell.GetLatticeType()]

    @property
    def lattice_params(self) -> np.ndarray[2, 3]:
        a = self.ob_unit_cell.GetA()
        b = self.ob_unit_cell.GetB()
        c = self.ob_unit_cell.GetC()
        alpha = self.ob_unit_cell.GetAlpha()
        beta = self.ob_unit_cell.GetBeta()
        gamma = self.ob_unit_cell.GetGamma()
        return np.array([[a, b, c], [alpha, beta, gamma]])

    @property
    def matrix(self) -> np.ndarray:
        ob_mat = self.ob_unit_cell.GetCellMatrix()
        return np.array([
            [ob_mat.Get(0, 0), ob_mat.Get(0, 1), ob_mat.Get(0, 2)],
            [ob_mat.Get(1, 0), ob_mat.Get(1, 1), ob_mat.Get(1, 2)],
            [ob_mat.Get(2, 0), ob_mat.Get(2, 1), ob_mat.Get(2, 2)]
        ])

    @staticmethod
    def matrix_to_params(matrix: np.ndarray):
        """ Covert the cell matrix to cell parameters: a, b, c, alpha, beta, gamma """
        va, vb, vc = matrix
        a = sum(va ** 2) ** 0.5
        b = sum(vb ** 2) ** 0.5
        c = sum(vc ** 2) ** 0.5

        alpha = np.arccos(np.dot(va, vb) / (a * b)) / np.pi * 180
        beta = np.arccos(np.dot(va, vc) / (a * c)) / np.pi * 180
        gamma = np.arccos(np.dot(vb, vc) / (b * c)) / np.pi * 180

        return a, b, c, alpha, beta, gamma

    @property
    def pack_molecule(self) -> "chem.Molecule":
        mol = self.molecule  # Get the contained Molecule

        if not mol:  # if you get None
            raise AttributeError("the crystal doesn't contain any Molecule!")

        ob_unit_cell = ob.OBUnitCell(self.ob_unit_cell)
        pack_mol = mol.copy()
        ob_unit_cell.FillUnitCell(pack_mol.ob_mol)  # Full the crystal

        pack_mol.build_bonds()
        pack_mol.assign_bond_types()

        pack_mol.reorder_ob_ids()  # Rearrange the atom indices and bond indices.

        return pack_mol

    def set_lattice(
            self,
            a: float, b: float, c: float,
            alpha: float, beta: float, gamma: float
    ):
        self.ob_unit_cell.SetData(a, b, c, alpha, beta, gamma)

    def set_vectors(
            self,
            va: Union[np.ndarray, Sequence],
            vb: Union[np.ndarray, Sequence],
            vc: Union[np.ndarray, Sequence]
    ):
        """"""
        vectors = [va, vb, vc]
        matrix = np.array(vectors)
        self.set_matrix(matrix)

    def set_matrix(self, matrix: np.ndarray):
        """ Set cell matrix for the crystal """
        if matrix.shape != (3, 3):
            raise AttributeError('the shape of cell_vectors should be [3, 3]')

        cell_params = map(float, self._matrix_to_params(matrix))

        self.ob_unit_cell.SetData(*cell_params)

    @property
    def space_group(self):
        space_group = self.ob_unit_cell.GetSpaceGroup()
        if space_group:
            return space_group.GetHMName()
        else:
            return None

    @space_group.setter
    def space_group(self, value: str):
        self._set_space_group(value)

    @property
    def volume(self):
        return self.ob_unit_cell.GetCellVolume()

    @property
    def vectors(self):
        v1, v2, v3 = self.ob_unit_cell.GetCellVectors()
        return np.array([
            [v1.GetX(), v1.GetY(), v1.GetZ()],
            [v2.GetX(), v2.GetY(), v2.GetZ()],
            [v3.GetX(), v3.GetY(), v3.GetZ()]
        ])


