"""
python v3.9.0
@Project: hotpot
@File   : math
@Auther : Zhiyuan Zhang
@Data   : 2024/12/18
@Time   : 16:32
"""
from typing import Optional
from itertools import combinations
import numpy as np


class Point:
    def __init__(self, x, y, z):
        self._pos = np.array([x, y, z])


def to_point(p):
    return np.array(p)


class Line:
    """"""
    def __init__(self, point1, point2):
        self.point1 = to_point(point1)
        self.point2 = to_point(point2)

    def __repr__(self):
        return f"Line({self.point1} + t({self.point2 - self.point1}))"

    @property
    def vector(self):
        return self.point2 - self.point1

    @property
    def identity_vector(self):
        """ return an identity vector """
        return (self.point2 - self.point1) / np.linalg.norm(self.point2 - self.point1)

    @property
    def segment_length(self) -> float:
        return np.linalg.norm(self.point2 - self.point1)

    @classmethod
    def from_vector(cls, point1, vector):
        return cls(point1, point1 + vector)

    def point_on_line(self, t, identity: bool = False):
        if identity:
            return self.point1 + t * self.identity_vector
        else:
            return self.point1 + t * self.vector

    def get_param_t(self, check_point):
        dir_vector_cpp1 = check_point - self.point1

        if round(len_vector_cpp1 := np.linalg.norm(dir_vector_cpp1), 8) == 0:
            return 0
        else:
            identity_vector_cpp1 = dir_vector_cpp1 / len_vector_cpp1
            if np.round(np.abs(np.dot(identity_vector_cpp1, self.identity_vector)), 8) != 1.0:
                raise AttributeError('the given point not on the line!!')

            return np.dot(dir_vector_cpp1, self.identity_vector) / self.segment_length


class Plane:
    """"""
    def __init__(self, p1, p2, p3):
        self.point1 = to_point(p1)
        self.point2 = to_point(p2)
        self.point3 = to_point(p3)
        if abs(np.dot(self.vector12, self.vector13)) < 1e-7:
            raise ValueError(f"The p1, p2, and p3 must not in a same line!\np1: {p1};\np2: {p2};\n p3: {p3}")

    def __repr__(self):
        a, b, c = self.identity_norm_vector
        x0, y0, z0 = self.point1
        return f"Plane({a}(x-{x0}) + {b}(y-{y0}) + {c}(z-{z0}) = 0)"

    @property
    def center(self):
        return (self.point1 + self.point2 + self.point3) / 3

    @property
    def vector12(self):
        return self.point2 - self.point1

    @property
    def vector13(self):
        return self.point3 - self.point1

    @property
    def vector23(self):
        return self.point3 - self.point2

    @property
    def norm_vector(self):
        return np.cross(self.vector12, self.vector13)

    @property
    def identity_norm_vector(self):
        return self.norm_vector / np.linalg.norm(self.norm_vector)

    @classmethod
    def from_vector(cls, point1, vector):
        # a(x-x1) + b(y-y1) + c(z-z1) = 0
        point2 = point1 + np.ones(3)
        point2[3] = np.divide(np.dot(point2[:2], vector[:2]), (-vector[2])) + point1[2]

        vector12 = point2 - point1
        point3 = np.cross(vector12, vector) + point1
        return cls(point1, point2, point3)

    def distance_with_point(self, point):
        return abs(np.dot(self.identity_norm_vector, point - self.point1))

    def is_on_plane(self, point, tol: float = 0.03):
        return self.distance_with_point(point) / (np.linalg.norm(point-self.point1) + 1e-6) < tol

    def is_line_intersect(self, line: Line):
        return np.dot(line.identity_vector, self.identity_norm_vector) != 0

    def line_intersect_point(self, line: Line) -> Optional:
        """
        Given a plane (123) and a line (AB) intersecting in a point p = pA + t(pB-pA),
        where the t is the parameter of line equation. The solution of the p equal to:

        t = <vec(n123),vec(A0)> / <vec(n123),vec(AB)>,
        p = p(t)

        Where vec(n123) is the normal vector of plane (123); vec(A0) is the vector from
        point pA in the line(AB) to the point p0 in the plane(123); vec(AB) is the vector
        from point pA in the line(AB) to point pB in the line(AB). <#, #> is the inner
        product operation.
        """
        if not self.is_line_intersect(line):
            return None

        #plane equation coefficients in point1
        # plane_coef = -np.dot(self.norm_vector, self.point1)

        # param in Line param equation
        # t = -plane_coef / np.dot(self.norm_vector, line.vector)
        t = np.dot(self.norm_vector, self.point1 - line.point1) / np.dot(self.norm_vector, line.vector)

        # Calculate intersect point by substitute t back the param equation of line
        return line.point_on_line(t)


def points_on_same_plane(*points):
    points = np.array(points)
    if points.shape[-1] != 3:
        raise AttributeError('The function just track points on 3 dimensions.')

    if len(points) < 3:
        return None
    if len(points) == 3:
        if np.linalg.det(points) < 1e-7:
            return None
        else:
            return True

    best_points = max(
        (list(p_indices) for p_indices in combinations(range(len(points)), 3)),
        key=lambda pi: np.linalg.det(points[pi])
    )

    try:
        plane = Plane(*points[best_points])
    except ValueError:
        return None

    return all(plane.is_on_plane(points[i]) for i in range(len(points)) if i not in best_points)


class CyclePlanes:
    def __init__(self, *points):
        self.points = to_point(points)

    def __len__(self):
        return len(self.points)

    def __repr__(self):
        return f"CyclePlanes({len(self.points)})"

    @property
    def planes(self) -> list[Plane]:
        center = self.center
        return [Plane(center, pi, pj) for pi, pj in zip(self.points[:-1], self.points[1:])]

    def get_plane(self, i: int) -> Plane:
        i, j = self._indices(i)
        return Plane(self.center, self.points[i], self.points[j])

    @property
    def norm_vector_degrees(self):
        ns = self.identity_norm_vectors
        degrees = []
        for n1, n2 in combinations(ns, 2):
            degrees.append(np.degrees(np.arccos(np.dot(n1, n2))))

        return degrees

    def _indices(self, i: int):
        if i == len(self.points) - 1:
            j = 0
        else:
            j = i + 1

        return i, j

    @property
    def max_norm_vector_degrees(self) -> float:
        return np.max(self.norm_vector_degrees)

    @property
    def mean_norm_vector_degrees(self) -> float:
        return np.mean(self.norm_vector_degrees)

    @property
    def center(self):
        return np.sum(self.points, axis=0) / len(self.points)

    @property
    def center_point_vectors(self):
        center = self.center
        return np.array([p - center for p in self.points])

    def edge_vector(self, i: int):
        i, j = self._indices(i)
        return self.points[j] - self.points[i]

    def center_point_vector(self, i: int):
        return self.points[i] - self.center

    def norm_vector(self, i: int):
        """"""
        i, j = self._indices(i)
        return np.cross(self.center_point_vector(i), self.center_point_vector(j))

    def identity_norm_vector(self, i):
        norm_vector = self.norm_vector(i)
        return norm_vector / np.linalg.norm(norm_vector)

    @property
    def norm_vectors(self):
        norm_vectors = []
        cp_vectors = self.center_point_vectors
        for i in range(len(self.points)):
            if i != len(self.points) - 1:
                norm_vectors.append(np.cross(cp_vectors[i], cp_vectors[i+1]))
            else:
                norm_vectors.append(np.cross(cp_vectors[i], cp_vectors[0]))

        return np.array(norm_vectors)

    @property
    def identity_norm_vectors(self):
        return np.array([v / np.linalg.norm(v) for v in self.norm_vectors])

    def point_in_which_edge_side(self, i: int, point) -> float:
        """
        Judge a given point in which side of a given edge (i), or exactly which side of a plane,
        which passes through the given edge (pi, pj) and is vertical to plane (center, pi, pj).

        If a point is on the plane, the function returns 0.
        If two points are on different sides of the plane, the return value have different sign (negative or positive).
        """
        return np.linalg.det([point - self.points[i], self.edge_vector(i), self.norm_vector(i)])

    def edge_center_side(self, i: int) -> float:
        """
        Judge the center point in which side of the plane,
        which passes through line (pi, pj) and is vertical to plane (center, pi, pj)
        """
        return self.point_in_which_edge_side(i, self.center)

    def line_intersect_points(self, line):
        intersect_points = []
        for plane in self.planes:
            intersect_points.append(plane.line_intersect_point(line))

        return intersect_points

    def in_same_side_with_center(self, i: int, point) -> bool:
        """ Judge the given point whether in a same side of a given edge (i, j) with center point """
        return self.point_in_which_edge_side(i, point) * self.edge_center_side(i) > 0

    def is_line_intersect_the_cycle(self, line: Line, segment: bool = True) -> bool:
        for i, plane in enumerate(self.planes):
            intersect_point = plane.line_intersect_point(line)
            if intersect_point is None:
                return False
            if not self.in_same_side_with_center(i, intersect_point):
                return False
            if segment and not(0 < line.get_param_t(intersect_point) < 1):
                return False

        return True
