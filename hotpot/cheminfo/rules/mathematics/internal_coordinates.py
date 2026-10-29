"""Existing internal-coordinate mathematics over explicit coordinate/edge arrays."""

from typing import Optional

import numpy as np

from hotpot.utils import types
from .graph.paths import depth_first_path


__all__ = ("InternalCoordinates",)


class InternalCoordinates:
    """
    Represents and manages internal coordinates of a molecular system.

    This class facilitates the representation of internal coordinates using methods such
    as z-matrices. It also provides various mathematical utilities for calculations
    related to molecular geometry, such as bond lengths, angles, and dihedrals. The
    internal structure is represented by graph theory concepts, and it supports
    operations on graph representations of molecular linkages.

    Attributes:
    coordinates (np.ndarray): The coordinates of the system's atoms.
    link_matrix (np.ndarray): The connectivity data of the internal structure.
    """
    def __init__(self, coordinates: np.ndarray, link_matrix: np.ndarray):
        self.coordinates = coordinates
        self.link_matrix = link_matrix

    @staticmethod
    def calc_zmat_idx(
            link_matrix: np.ndarray,
            node_count: Optional[int] = None,
    ) -> np.ndarray:
        """
            Calculate a Z-Matrix index from an explicit integer edge array.

            Parameters:
            link_matrix: A numpy array of shape ``(n_edges, 2)``.
            node_count: Optional total node count when isolated nodes are present.

            Returns:
            np.ndarray: A numpy array of shape (n_nodes, 4) where each node's connectivity pattern is
                described by its corresponding index path in the graph.

            Raises:
            RuntimeError: If an incorrect Z-Matrix is constructed due to issues in graph traversal.
        """
        edges = tuple(tuple(map(int, edge)) for edge in link_matrix)
        if node_count is None:
            node_count = 1 + max(
                (node for edge in edges for node in edge),
                default=-1,
            )
        adjacency = [set() for _ in range(node_count)]
        for first, second in edges:
            adjacency[first].add(second)
            adjacency[second].add(first)

        zmat_idx = -1 * np.ones((node_count, 4), dtype=int)
        visited_nodes = set()
        node = 0
        while len(visited_nodes) < node_count:
            row = len(visited_nodes)
            visited_nodes.add(node)

            path = depth_first_path(
                node_count,
                edges,
                start_node=node,
                scope_nodes=visited_nodes,
                max_depth=min(row + 1, 4),
            )
            zmat_idx[row, :row+1] = path

            try:
                node = min(
                    neighbor
                    for visited in visited_nodes
                    for neighbor in adjacency[visited]
                    if neighbor not in visited_nodes
                )
            except ValueError:
                if len(visited_nodes) != node_count:
                    raise RuntimeError("Get an incorrect Z-Matirx !!")

        return zmat_idx

    @staticmethod
    def calc_zmat(link_matrix: np.ndarray, coords: np.ndarray) -> np.ndarray:
        """
        Compute the Z-matrix from an explicit edge array and coordinates.

        Args:
            link_matrix: A numpy array of shape ``(n_edges, 2)``.
            coords (np.ndarray): A 2D NumPy array of atomic coordinates with shape (N, 3), 
                where N is the number of atoms.

        Returns:
            np.ndarray: A 2D NumPy array representing the Z-matrix. Each row corresponds to an
                atom in the molecule, and the columns contain the bond length, bond angle, 
                and dihedral angle, respectively, calculated for that atom using its connectivity.
        """
        zmat_idx = InternalCoordinates.calc_zmat_idx(
            link_matrix,
            node_count=len(coords),
        )
        zmat = -1 * np.ones((zmat_idx.shape[0], 3))

        for i, index in enumerate(zmat_idx):
            zmat[i, :i] = InternalCoordinates._calc_zmat_line(index, coords)

        return zmat

    @staticmethod
    def _calc_zmat_line(index: types.ArrayLike, coords: np.ndarray):
        """
            Calculates the Z-matrix parameters for a given set of indices and coordinates.

            This method computes bond length, bond angle, and dihedral angle depending 
            on the number of vectors derived from the input indices. It uses `numpy`
            for vector operations such as calculating norms, dot products, and cross 
            products. Angles are converted from radians to degrees. The resulting 
            parameters represent molecular geometry with respect to the provided indices.

            Args:
                index (types.ArrayLike): Array-like object containing indices that specify 
                atoms connected to the current atom. It must have a length of 4.

                coords (np.ndarray): A 2D numpy array where each row represents the

            Returns:
                list: A list of calculated Z-matrix parameters containing bond length, 
                bond angle, and optionally dihedral angle as applicable.
        """
        assert len(index) == 4

        vectors = [coords[index[i]] - coords[index[i-1]] for i in range(1, 4) if index[i] >= 0]

        results = []
        if len(vectors) >= 1:
            r = np.linalg.norm(vectors[0])
            results.append(r)

        if len(vectors) >= 2:
            r1 = np.linalg.norm(vectors[1])
            cos_a = np.dot(vectors[0], vectors[1]) / (r*r1)
            a = np.degrees(np.arccos(cos_a))
            results.append(a)

        if len(vectors) == 3:
            n1 = np.cross(vectors[0], vectors[1])
            n2 = np.cross(vectors[1], vectors[2])

            norm_n1 = np.linalg.norm(n1)
            norm_n2 = np.linalg.norm(n2)

            if norm_n1 == 0 or norm_n2 == 0:
                cos_dehedral = -1
            else:
                cos_dehedral = round(np.dot(n1, n2) / (norm_n1 * norm_n2), 8)

            dehedral = np.degrees(np.arccos(cos_dehedral))
            results.append(dehedral)

        return results

    @staticmethod
    def calc_angle(a: types.ArrayLike, b: types.ArrayLike, c: types.ArrayLike):
        """
            Calculates the angle defined by three points.

            This method computes the angle in radians at the middle point
            defined by the lines connecting three points `a`, `b`, and `c`.
            The calculation assumes the three points form two vectors
            emanating from the middle point, and the angle is evaluated
            between these two vectors.

            Args:
                a (types.ArrayLike): The coordinates of the first point.
                b (types.ArrayLike): The coordinates of the middle point.
                c (types.ArrayLike): The coordinates of the third point.

            Returns:
                float: The angle in radians between the vectors defined by
                the points.

            Raises:
                ValueError: If the input arrays are of incompatible shapes
                or do not represent valid points.
        """
        return InternalCoordinates._calc_angle(np.array(a), np.array(b), np.array(c))

    @staticmethod
    def _calc_angle(a: np.ndarray, b: np.ndarray, c: np.ndarray, return_degree: bool = True):
        """
            Calculate the angle formed by three points a, b, and c in a two-dimensional space. 
            The angle is computed using the dot product formula and can be returned in radians 
            or degrees.

            Args:
                a (np.ndarray): The first point in the 2D space.
                b (np.ndarray): The second point in the 2D space.
                c (np.ndarray): The third point in the 2D space.
                return_degree (bool): Whether to return the angle in degrees or radians. 
                    Defaults to True (degrees).

            Returns:
                float: The angle between the vectors ab and bc, in degrees or radians 
                depending on return_degree.

            Raises:
                None: This method does not raise exceptions.
        """
        # Calculate vectors ab and bc
        ab = b - a
        bc = c - b

        # Calculate the dot product and magnitudes
        abc_dot = np.dot(ab, bc)
        norm_ab = np.linalg.norm(ab)
        norm_bc = np.linalg.norm(bc)

        # Calculate the angle in radians
        cos_theta = abc_dot / (norm_ab * norm_bc)
        angle_radians = np.arccos(cos_theta)

        if return_degree:
            return np.degrees(angle_radians) # Convert to degrees
        else:
            return angle_radians

    @staticmethod
    def calc_dehedral(
            a: types.ArrayLike,
            b: types.ArrayLike,
            c: types.ArrayLike,
            d: types.ArrayLike,
            return_degree: bool = True
    ):
        """
        Calculate the dihedral angle formed by four points.

        This static method computes the dihedral angle, measured in degrees or radians,
        for four consecutive points in space. The input points are specified as array-like
        structures. The computation determines the angle formed by the planes defined
        by the four points.

        Parameters:
            a (ArrayLike): The first point in the sequence, given in Cartesian coordinates.
            b (ArrayLike): The second point in the sequence, given in Cartesian coordinates.
            c (ArrayLike): The third point in the sequence, given in Cartesian coordinates.
            d (ArrayLike): The fourth point in the sequence, given in Cartesian coordinates.
            return_degree (bool, optional): Determines the unit of the returned angle.
                If True, the angle is returned in degrees. If False, the angle is returned
                in radians. Defaults to True.

        Returns:
            float: The computed dihedral angle in degrees or radians, based on the value
                of the return_degree parameter.
        """
        return InternalCoordinates._calc_dehedral(np.array(a), np.array(b), np.array(c), np.array(d), return_degree)

    @staticmethod
    def _calc_dehedral(a: np.ndarray, b: np.ndarray, c: np.ndarray, d: np.ndarray, return_degree: bool = True):
        """
            Calculates the dehedral angle formed by four points in 3D space.

            This method computes the dehedral angle between the planes determined by 
            four consecutive points (A, B, C, D). The calculation is performed using 
            the vector cross products of the line segments between these points. The 
            result can be returned in either degrees or radians as specified.

            Args:
                a (np.ndarray): Coordinates of the first point (A), given as a 
                    1-dimensional numpy array.
                b (np.ndarray): Coordinates of the second point (B), given as a 
                    1-dimensional numpy array.
                c (np.ndarray): Coordinates of the third point (C), given as a 
                    1-dimensional numpy array.
                d (np.ndarray): Coordinates of the fourth point (D), given as a 
                    1-dimensional numpy array.
                return_degree (bool): Boolean flag indicating whether the angle should 
                    be returned in degrees (True) or in radians (False). Default is True.

            Returns:
                float: The calculated dehedral angle in degrees (when `return_degree` 
                is True) or radians (when `return_degree` is False).
        """
        # Calculate vectors ab, bc, and cd
        ab = b - a
        bc = c - b
        cd = d - c

        n1 = np.cross(ab, bc)
        n2 = np.cross(bc, cd)

        norm_n1 = np.linalg.norm(n1)
        norm_n2 = np.linalg.norm(n2)

        if norm_n1 == 0 or norm_n2 == 0:
            cos_dehedral = -1
        else:
            cos_dehedral = np.dot(n1, n2) / (norm_n1*norm_n2)

        dehedral_radians = np.arccos(cos_dehedral)
        if return_degree:
            return np.degrees(dehedral_radians) # Convert to degrees
        else:
            return dehedral_radians
