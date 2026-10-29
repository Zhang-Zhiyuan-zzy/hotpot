"""Numerical view calculations over explicit rows, edges and coordinates."""

from itertools import combinations, product

import numpy as np
from scipy.spatial.distance import pdist, squareform


__all__ = ()


def _angle_rows(adjacency):
    return tuple((first, center, second) for center, neighbors in enumerate(adjacency)
                 for first, second in combinations(neighbors, 2))


def _torsion_rows(edges, adjacency):
    return tuple((first, left, right, last) for left, right in edges
                 for first, last in product(
                     (row for row in adjacency[left] if row != right),
                     (row for row in adjacency[right] if row != left)))


def _pair_distance(first, second):
    return np.linalg.norm(second - first)


def _pair_distances(coordinates):
    return pdist(coordinates)


def _distance_matrix(coordinates):
    return squareform(pdist(coordinates))
