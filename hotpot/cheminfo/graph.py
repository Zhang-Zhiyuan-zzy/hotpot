"""
python v3.9.0
@Project: hotpot
@File   : graph
@Auther : Zhiyuan Zhang
@Data   : 2024/8/23
@Time   : 15:06
"""
from typing import *
import numpy as np
import networkx as nx

from hotpot.utils import types


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


def linkmat2adj(note_num: int, linkmat: np.ndarray) -> np.ndarray:
    """ Convert the link matrix with shape (BN, 2) to an adjacency matrix with shape of (AN, AN) """
    if note_num <= 0 or linkmat.size == 0:
        return np.zeros((1, 0), dtype=int)

    assert len(linkmat.shape) == 2
    assert linkmat.shape[1] == 2

    adj = np.zeros((note_num, note_num))
    adj[linkmat[:, 0], linkmat[:, 1]] = 1
    adj[linkmat[:, 1], linkmat[:, 0]] = 1

    return adj

def adj2laplacian(adj: np.ndarray, norm: bool = True) -> np.ndarray:
    """
    convert adjacency matrix to laplacian matrix
    Args:
        adj: adjacency matrix
        norm: whether to return normalized laplacian matrix

    Return:
         Laplacian matrix or normalized Laplacian matrix
    """
    if adj.size == 0:
        return np.zeros((1, 0), dtype=int)

    deg = np.sum(adj, axis=1)
    eye = np.eye(adj.shape[0])

    lap = np.diag(deg) - adj
    if norm:
        root_deg = np.sqrt(deg)
        root_deg[root_deg == 0] = np.inf
        lap_row = (lap / root_deg).T
        norm_lap = (lap_row / root_deg).T
        return norm_lap
        # return eye - np.linalg.inv(deg ** 0.5) @ adj @ np.linalg.inv(deg ** 0.5)
    else:
        return lap


def _spectrum_sort(spectrum: np.ndarray) -> np.ndarray:
    """ sort spectrum values according to its absolute values """
    spectrum = np.sort(spectrum)[::-1]
    sorted_idx = np.argsort(np.abs(spectrum))[::-1]
    return spectrum[sorted_idx]


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


def graph_dfs_path(
        graph: nx.Graph,
        start_node: int = None,
        scope_nodes: Container = None,
        min_deep: int = None,
        max_deep: int = None
) -> Optional[list[int]]:
    """"""
    def _dfs(_node: int, visited: list[int]):
        visited.append(_node)
        if max_deep and len(visited) >= max_deep:
            return visited

        for child in nx.neighbors(graph, _node):
            if (child not in visited) and (scope_nodes and child in scope_nodes):
                return _dfs(child, visited)

        if min_deep and len(visited) >= min_deep:
            return visited

    if start_node is None:
        start_node = 0

    return _dfs(start_node, [])


def graph_dfs_paths(
        graph: nx.Graph,
        start_node: int,
        scope_nodes: Container = None,
        min_deep: int = None,
        max_deep: int = None
) -> list[list[int]]:
    paths = []

    def _dfs(node: int, visited: set[int], path: list[int]) -> None:
        path.append(node)
        visited.add(node)

        if max_deep and len(visited) >= max_deep:
            paths.append(path)
            return

        for child in nx.neighbors(graph, node):
            if (child not in visited) and (scope_nodes and child in scope_nodes):
                _dfs(child, visited, path)

        if min_deep and len(visited) >= min_deep:
            paths.append(path)

    if start_node is None:
        start_node = 0

    return paths


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



