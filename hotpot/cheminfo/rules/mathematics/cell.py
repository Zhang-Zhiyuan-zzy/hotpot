"""Cell parameter conversion preserving Hotpot’s characterized row convention."""

import numpy as np


__all__ = ()


def _matrix_to_params(matrix):
    """ Covert the cell matrix to cell parameters: a, b, c, alpha, beta, gamma """
    va, vb, vc = matrix
    a = sum(va ** 2) ** 0.5
    b = sum(vb ** 2) ** 0.5
    c = sum(vc ** 2) ** 0.5

    alpha = np.arccos(np.dot(va, vb) / (a * b)) / np.pi * 180
    beta = np.arccos(np.dot(va, vc) / (a * c)) / np.pi * 180
    gamma = np.arccos(np.dot(vb, vc) / (b * c)) / np.pi * 180

    return a, b, c, alpha, beta, gamma
