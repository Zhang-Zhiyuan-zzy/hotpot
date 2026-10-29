"""Conversions between graph matrix representations."""

import numpy as np


__all__ = ("linkmat2adj", "adj2laplacian")


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
