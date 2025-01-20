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
        self.mol = None
        self._obcell = None

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

    @classmethod
    def from_matrix(cls, matrix: np.ndarray):
        return cls(*cls._matrix_to_params(matrix))

    @property
    def obcell(self):
        obcell = ob.OBUnitCell()
        obcell.SetData(self.a, self.b, self.c, self.alpha, self.beta, self.gamma)
        return obcell

    @property
    def space_group(self):
        return NotImplementedError
