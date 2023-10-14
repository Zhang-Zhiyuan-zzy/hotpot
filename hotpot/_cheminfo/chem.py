"""
python v3.9.0
@Project: hotpot
@File   : molecule
@Auther : Zhiyuan Zhang
@Data   : 2023/10/14
@Time   : 16:25
"""
from abc import ABC

from openbabel import openbabel as ob, pybel as pb

from ._base import Wrapper


class Molecule(Wrapper, ABC):
    """Represent an intuitive molecule"""
    def __init__(self, ob_mol):
        super().__init__(ob_mol)

    @property
    def ob_mol(self) -> ob.OBMol:
        return self._obj


