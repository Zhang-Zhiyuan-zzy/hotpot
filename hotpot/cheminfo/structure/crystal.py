"""Six writable cell parameters, independent of external toolkit state."""
import numpy as np


class Crystal:
    def __init__(self, a, b, c, alpha, beta, gamma):
        self.a = a
        self.b = b
        self.c = c
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

    @staticmethod
    def _matrix_to_params(matrix):
        from ..rules.mathematics.cell import _matrix_to_params

        return _matrix_to_params(matrix)

    @classmethod
    def from_matrix(cls, matrix: np.ndarray):
        return cls(*cls._matrix_to_params(matrix))
