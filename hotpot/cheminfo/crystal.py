import numpy as np
from openbabel import openbabel as ob


class Crystal:
    def __init__(self, a, b, c, alpha, beta, gamma, *, mol=None):
        self.a = a
        self.b = b
        self.c = c
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.mol = mol
        self._obcell = None

    @staticmethod
    def _matrix_to_params(matrix: np.ndarray):
        from .rules.mathematics.cell import _matrix_to_params
        return _matrix_to_params(matrix)

    @classmethod
    def from_matrix(cls, matrix: np.ndarray):
        return cls(*cls._matrix_to_params(matrix))

    @property
    def obcell(self) -> ob.OBUnitCell:
        obcell = ob.OBUnitCell()
        obcell.SetData(self.a, self.b, self.c, self.alpha, self.beta, self.gamma)
        return obcell
