"""Electronic-configuration augmented graph spectra."""

from typing import Literal

import numpy as np

from hotpot.utils import types

from .matrix import adj2laplacian


__all__ = (
    "calc_electron_config",
    "atoms_electron_configurations",
    "calc_spectrum",
    "GraphSpectrum",
)


def calc_electron_config(atomic_number: int, length: int = 4) -> (int, list):
    shells = [
        [2],
        [2, 6],
        [2, 6],
        [2, 10, 6],
        [2, 10, 6],
        [2, 14, 10, 6],
        [2, 14, 10, 6],
        [2, 18, 14, 10, 6],
    ]
    conf = []
    _atomic_number = atomic_number

    n = 0
    l = 0
    while _atomic_number > 0:
        if l >= len(shells[n]):
            n += 1
            l = 0
            conf = []

        if _atomic_number - shells[n][l] > 0:
            conf.append(shells[n][l])
        else:
            conf.append(_atomic_number)

        _atomic_number -= shells[n][l]
        l += 1

    return n, conf + [0] * (length - len(conf))


def atoms_electron_configurations(
        atomic_numbers: np.ndarray,
        length: int = 4
) -> np.ndarray:
    atomic_numbers = np.array(atomic_numbers)

    confs = []
    for atomic_number in atomic_numbers:
        n, conf = calc_electron_config(int(atomic_number), length=length)
        confs.append([n] + conf)

    return np.array(confs).T


def _spectrum_sort(spectrum: np.ndarray) -> np.ndarray:
    """ sort spectrum values according to its absolute values """
    # TODO: Fix zero-padding disturbing physical interpretation.
    # Candidates for future refactoring:
    # 1. [Best] Wasserstein Distance (Scipy): Natural transport cost, scale-invariant.
    # 2. [Fast] Spectral Histogram/Binning: Converts spectrum to fixed-size density vector.
    # 3. [Smooth] Kernel Density Estimation (KDE): Smooths eigenvalues into continuous function.
    # TODO End.
    spectrum = np.round(spectrum, 8)
    return np.sort(spectrum)[::-1]
    # TODO: discarded, old measure
    # spectrum = np.sort(spectrum)[::-1]
    # sorted_idx = np.argsort(np.abs(spectrum))[::-1]
    # return spectrum[sorted_idx]


def calc_spectrum(adj: types.ArrayLike, atomic_numbers: types.ArrayLike, length: int = 4) -> np.ndarray:
    """
    Calculate the spectrum matrix of a given molecule defined by an adjacency matrix and atomic numbers 1D array
    Args:
        adj (np.ndarray): a square matrix of adjacency
        atomic_numbers (np.ndarray): a 1D array of atomic numbers, with a same order with the adjacency matrix
        length (int): the default length of electric configurations
    """
    adj = np.array(adj, dtype=int)
    atomic_numbers = np.array(atomic_numbers, dtype=int).flatten()

    assert len(adj.shape) == 2
    assert adj.shape[0] == adj.shape[1] == len(atomic_numbers)

    spectrum = []

    # Add normalize Laplacian matrix
    spectrum.append(_spectrum_sort(np.linalg.eigvals(adj2laplacian(adj, norm=True)).real))

    # add electric configurations filled adjacency matrix
    confs = atoms_electron_configurations(atomic_numbers, length=length)
    for c in confs:
        spectrum.append(_spectrum_sort(np.linalg.eigvals(np.diag(c) + adj).real))

    return np.array(spectrum)


class GraphSpectrum:
    def __init__(
            self,
            spectrum: np.ndarray,
            norm: Literal['infinite', 'min', 'l1', 'l2'] = 'l2'
    ):
        self.spectrum = spectrum
        self.norm = norm

    def __or__(self, other: "GraphSpectrum"):
        return self.similarity(other)

    @property
    def vectors(self):
        """ Just an alias of spectrum """
        return self.spectrum

    def similarity(self, other: "GraphSpectrum"):
        """"""
        if self.width >= other.width:
            vct1 = self.spectrum
            vct2 = other.spectrum
        else:
            vct1 = other.spectrum
            vct2 = self.spectrum

        if vct1.shape[1] != vct2.shape[1]:
            vct2 = np.pad(vct2, ((0, 0), (0, vct1.shape[1] - vct2.shape[1])))

        dot = np.diag(np.dot(vct1, vct2.T))
        norm1 = np.linalg.norm(vct1, axis=1)
        norm2 = np.linalg.norm(vct2, axis=1)

        vector = dot / (norm1 * norm2)

        if self.norm == 'l2':
            return np.linalg.norm(vector) / np.sqrt(len(vector))
        elif self.norm == 'infinite':
            return np.max(vector)
        elif self.norm == 'min':
            return np.min(vector)
        elif self.norm == 'l1':
            return sum(vector) / len(vector)

        # TODO: Discarded later
        return dot / (norm1 * norm2)

    @classmethod
    def from_adj_atoms(
            cls, adj: np.ndarray,
            atomic_numbers: np.ndarray, length: int = 4,
            norm: Literal['infinite', 'min', 'l1', 'l2'] = 'l2'
    ) -> "GraphSpectrum":
        """
        Generate a GraphSpectrum of a molecule by given adjacency matrix and atomic numbers.
        Args:
            adj (np.ndarray): a square matrix of adjacency
            atomic_numbers (np.ndarray): a 1D array of atomic numbers, with a same order with the adjacency matrix
            length (int): the default length of electric configurations
            norm (str, optional): how to calculate the norm of spectrum
        """
        return cls(calc_spectrum(adj, atomic_numbers, length), norm)

    @property
    def width(self) -> int:
        return self.spectrum.shape[1]
