"""
python v3.9.0
@Project: hotpot
@File   : math
@Auther : Zhiyuan Zhang
@Data   : 2024/12/18
@Time   : 16:32
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
from enum import Enum
from typing import Any, Literal, Mapping, Optional, Sequence, Tuple, Union
from itertools import combinations
import numpy as np


PairScope = Literal["all", "bonded", "nonbonded"]
RingScope = Literal["full_graph", "ligand_skeleton"]
QualityLevel = Literal["off", "basic", "standard", "strict"]
ForceFieldStage = Literal["candidate", "final"]


@dataclass(frozen=True)
class AtomPairGeometryIssue:
    """A distance-based geometry problem involving two atoms."""

    kind: Literal["overlap", "too_close", "short_bond"]
    atom_indices: Tuple[int, int]
    distance: float
    threshold: float


@dataclass(frozen=True)
class GeometryCheck:
    """One serializable result in a geometry-quality report."""

    name: str
    passed: bool
    severity: Literal["info", "warning", "error"] = "error"
    measured: Any = None
    threshold: Any = None
    atom_indices: Tuple[int, ...] = ()
    bond_indices: Tuple[int, ...] = ()
    message: str = ""


@dataclass(frozen=True)
class GeometryQualityThresholds:
    """Numerical thresholds used by :func:`evaluate_geometry_quality`."""

    overlap_tolerance: float = 1.0e-3
    basic_minimum_distance: float = 0.40
    standard_minimum_distance: float = 0.50
    standard_covalent_radius_scale: float = 0.55
    maximum_bond_distance: float = 30.0
    covalent_bond_ratio: Tuple[float, float] = (0.65, 1.45)
    metal_ligand_bond_ratio: Tuple[float, float] = (0.65, 1.60)
    strict_rms_gradient: float = 1.0
    strict_max_gradient: float = 5.0
    strict_energy_change: float = 1.0e-4
    strict_max_displacement: float = 1.0e-4
    strict_stability_window: int = 5


@dataclass(frozen=True)
class AtomTopologySignature:
    """Stable identity and chemistry for an atom present before optimization."""

    index: int
    atom_id: int
    atomic_number: int
    formal_charge: int


@dataclass(frozen=True)
class BondTopologySignature:
    """Stable topology for a bond present before optimization."""

    atom_indices: Tuple[int, int]
    bond_order: float
    bond_kind: str


@dataclass(frozen=True)
class TopologyReference:
    """Immutable topology snapshot accepted by the geometry-quality gate."""

    atoms: Tuple[AtomTopologySignature, ...]
    bonds: Tuple[BondTopologySignature, ...]
    allow_added_hydrogens: bool = True


@dataclass(frozen=True)
class GeometryQualityReport:
    """Structured result returned by the geometry-quality gate."""

    level: QualityLevel
    passed: bool
    checks: Tuple[GeometryCheck, ...]
    metrics: Mapping[str, Any] = field(default_factory=dict)

    @property
    def failures(self) -> Tuple[GeometryCheck, ...]:
        return tuple(
            check for check in self.checks
            if not check.passed and check.severity == "error"
        )

    @property
    def warnings(self) -> Tuple[GeometryCheck, ...]:
        return tuple(
            check for check in self.checks
            if not check.passed and check.severity == "warning"
        )

    def to_dict(self) -> dict:
        """Return a JSON-serializable representation of the report."""
        return asdict(self)


class Point:
    def __init__(self, x, y, z):
        self._pos = np.array([x, y, z])


def to_point(p):
    return np.array(p)


class LinesRelationship(Enum):
    INTERSECT = "intersect"
    PARALLEL = "parallel"
    SKEW = "skew"


def get_line_relationship(v1: np.ndarray, v2: np.ndarray, p1: np.ndarray, p2: np.ndarray) -> str:
    """
    Determines the relationship between two lines in 3D space.

    The lines are defined by a point and a direction vector.
    Line 1: r = p1 + t * v1
    Line 2: r = p2 + s * v2

    Args:
        v1 (np.ndarray): Direction vector of the first line.
        v2 (np.ndarray): Direction vector of the second line.
        p1 (np.ndarray): A point on the first line.
        p2 (np.ndarray): A point on the second line.

    Returns:
        str: The relationship: "parallel", "intersecting", or "skew".
    """
    # Check for parallelism by computing the cross product of direction vectors.
    # If the cross product is a zero vector, the vectors are collinear, so lines are parallel.
    cross_v = np.cross(v1, v2)
    if np.allclose(cross_v, [0, 0, 0]):
        return LinesRelationship.PARALLEL

    # Check for intersection or skewness using the scalar triple product.
    # This checks if the vectors v1, v2, and (p2 - p1) are coplanar.
    p1p2 = p2 - p1
    scalar_triple_product = np.dot(cross_v, p1p2)

    if np.isclose(scalar_triple_product, 0):
        return LinesRelationship.INTERSECT
    else:
        return LinesRelationship.SKEW

def calculate_line_distance(v1: np.ndarray, v2: np.ndarray, p1: np.ndarray, p2: np.ndarray) -> (str, float):
    """
    Calculates the shortest distance between two lines in 3D space.

    The function first determines the relationship between the lines (parallel,
    intersecting, or skew) and then applies the appropriate formula.

    Args:
        v1 (np.ndarray): Direction vector of the first line.
        v2 (np.ndarray): Direction vector of the second line.
        p1 (np.ndarray): A point on the first line.
        p2 (np.ndarray): A point on the second line.

    Returns:
        Tuple[str, float]: A tuple containing the relationship type and the
                           calculated shortest distance.
    """
    relationship = get_line_relationship(v1, v2, p1, p2)
    p1p2 = p2 - p1

    if relationship == LinesRelationship.INTERSECT:
        # The distance between intersecting lines is 0.
        return relationship, 0.0

    elif relationship == LinesRelationship.PARALLEL:
        # Formula for parallel lines: ||(p2-p1) x v1|| / ||v1||
        distance = np.linalg.norm(np.cross(p1p2, v1)) / np.linalg.norm(v1)
        return relationship, float(distance)

    elif relationship == LinesRelationship.SKEW:
        # Formula for skew lines: |(v1 x v2) . (p2-p1)| / ||v1 x v2||
        cross_v = np.cross(v1, v2)
        distance = np.abs(np.dot(cross_v, p1p2)) / np.linalg.norm(cross_v)
        return relationship, float(distance)

    else:
        raise RuntimeError(f"Unknown relationship: {relationship}")


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

    def distance_to_line(self, other: "Line") -> (str, float):
        sp = self.point1
        op = other.point1 if not np.isclose(sp, other.point1).all() else other.point2

        rela, distance = calculate_line_distance(self.vector, other.vector, sp, op)
        return rela, distance


class Plane:
    """"""
    def __init__(self, p1, p2, p3):
        self.point1 = to_point(p1)
        self.point2 = to_point(p2)
        self.point3 = to_point(p3)
        if np.linalg.norm(np.cross(self.vector12, self.vector13)) < 1e-7:
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
        point1 = to_point(point1)
        vector = to_point(vector)
        axis = np.zeros(3)
        axis[int(np.argmin(np.abs(vector)))] = 1.0
        vector12 = np.cross(vector, axis)
        vector13 = np.cross(vector, vector12)
        return cls(point1, point1 + vector12, point1 + vector13)

    def distance_with_point(self, point):
        return abs(np.dot(self.identity_norm_vector, point - self.point1))

    def is_on_plane(self, point, tol: float = 0.03):
        return self.distance_with_point(point) / (np.linalg.norm(point-self.point1) + 1e-6) < tol

    def is_line_intersect(self, line: Line):
        return not np.isclose(
            np.dot(line.identity_vector, self.identity_norm_vector),
            0.0,
        )

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
        normal = np.cross(points[1] - points[0], points[2] - points[0])
        return None if np.linalg.norm(normal) < 1.0e-7 else True

    best_points = max(
        (list(p_indices) for p_indices in combinations(range(len(points)), 3)),
        key=lambda indices: np.linalg.norm(np.cross(
            points[indices[1]] - points[indices[0]],
            points[indices[2]] - points[indices[0]],
        )),
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
        return [self.get_plane(i) for i in range(len(self.points))]

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
        return _line_intersects_polygon(
            self.points,
            np.asarray(line.point1, dtype=float),
            np.asarray(line.point2, dtype=float),
            tolerance=1.0e-8,
            segment=segment,
        )


@dataclass(frozen=True)
class _AtomPairTable:
    atoms: Tuple[Any, ...]
    atom_indices: Tuple[int, ...]
    first: np.ndarray
    second: np.ndarray
    distances: np.ndarray
    bonded_pairs: frozenset


def _atom_coordinates(atoms: Sequence[Any]) -> np.ndarray:
    if not atoms:
        return np.empty((0, 3), dtype=float)
    return np.asarray([atom.coordinates for atom in atoms], dtype=float)


def _atom_index(atom: Any, fallback: int) -> int:
    return int(getattr(atom, "idx", fallback))


def _pair_table(obj: Any, coordinates: Optional[np.ndarray] = None) -> _AtomPairTable:
    atoms = tuple(obj.atoms)
    atom_indices = tuple(_atom_index(atom, i) for i, atom in enumerate(atoms))
    coords = _atom_coordinates(atoms) if coordinates is None else coordinates
    first, second = np.triu_indices(len(atoms), 1)
    distances = np.linalg.norm(coords[first] - coords[second], axis=1)
    positions = {id(atom): i for i, atom in enumerate(atoms)}
    bonded_pairs = frozenset(
        tuple(sorted((positions[id(bond.atom1)], positions[id(bond.atom2)])))
        for bond in obj.bonds
        if id(bond.atom1) in positions and id(bond.atom2) in positions
    )
    return _AtomPairTable(
        atoms=atoms,
        atom_indices=atom_indices,
        first=first,
        second=second,
        distances=distances,
        bonded_pairs=bonded_pairs,
    )


def _pair_scope_mask(table: _AtomPairTable, pair_scope: PairScope) -> np.ndarray:
    if pair_scope not in ("all", "bonded", "nonbonded"):
        raise ValueError(f"Unknown atom-pair scope: {pair_scope!r}")
    if pair_scope == "all":
        return np.ones(len(table.distances), dtype=bool)

    bonded = np.fromiter(
        (
            (int(i), int(j)) in table.bonded_pairs
            for i, j in zip(table.first, table.second)
        ),
        dtype=bool,
        count=len(table.distances),
    )
    return bonded if pair_scope == "bonded" else ~bonded


def _overlap_issues(
    table: _AtomPairTable,
    tolerance: float,
) -> Tuple[AtomPairGeometryIssue, ...]:
    return tuple(_iter_overlap_issues(table, tolerance))


def _iter_overlap_issues(
    table: _AtomPairTable,
    tolerance: float,
):
    for position in np.flatnonzero(table.distances <= tolerance):
        first = int(table.first[position])
        second = int(table.second[position])
        yield AtomPairGeometryIssue(
            kind="overlap",
            atom_indices=(table.atom_indices[first], table.atom_indices[second]),
            distance=float(table.distances[position]),
            threshold=float(tolerance),
        )


def find_overlapping_atom_pairs(
        mol: Any,
        *,
        tolerance: float = 1.0e-3,
) -> Tuple[AtomPairGeometryIssue, ...]:
    """Return atom pairs whose coordinates coincide within ``tolerance``."""
    return _overlap_issues(_pair_table(mol), tolerance)


def has_overlapping_atoms(mol: Any, *, tolerance: float = 1.0e-3) -> bool:
    """Return whether any two atoms overlap within ``tolerance``."""
    return any(_iter_overlap_issues(_pair_table(mol), tolerance))


def _too_close_issues(
        table: _AtomPairTable,
        *,
        minimum_distance: float,
        covalent_radius_scale: Optional[float],
        pair_scope: PairScope,
        include_overlaps: bool,
        overlap_tolerance: float,
) -> Tuple[AtomPairGeometryIssue, ...]:
    return tuple(_iter_too_close_issues(
        table,
        minimum_distance=minimum_distance,
        covalent_radius_scale=covalent_radius_scale,
        pair_scope=pair_scope,
        include_overlaps=include_overlaps,
        overlap_tolerance=overlap_tolerance,
    ))


def _iter_too_close_issues(
    table: _AtomPairTable,
    *,
    minimum_distance: float,
    covalent_radius_scale: Optional[float],
    pair_scope: PairScope,
    include_overlaps: bool,
    overlap_tolerance: float,
):
    thresholds = np.full(len(table.distances), minimum_distance, dtype=float)
    if covalent_radius_scale is not None:
        radii = np.asarray(
            [float(atom.covalent_radius) for atom in table.atoms],
            dtype=float,
        )
        scaled = covalent_radius_scale * (
            radii[table.first] + radii[table.second]
        )
        thresholds = np.maximum(thresholds, scaled)

    mask = _pair_scope_mask(table, pair_scope) & (table.distances < thresholds)
    if not include_overlaps:
        mask &= table.distances > overlap_tolerance

    for position in np.flatnonzero(mask):
        first = int(table.first[position])
        second = int(table.second[position])
        yield AtomPairGeometryIssue(
            kind="too_close",
            atom_indices=(table.atom_indices[first], table.atom_indices[second]),
            distance=float(table.distances[position]),
            threshold=float(thresholds[position]),
        )


def find_too_close_atom_pairs(
        mol: Any,
        *,
        minimum_distance: float = 0.50,
        covalent_radius_scale: Optional[float] = None,
        pair_scope: PairScope = "all",
        include_overlaps: bool = True,
        overlap_tolerance: float = 1.0e-3,
) -> Tuple[AtomPairGeometryIssue, ...]:
    """Return atom pairs below an absolute or radius-scaled separation."""
    return _too_close_issues(
        _pair_table(mol),
        minimum_distance=minimum_distance,
        covalent_radius_scale=covalent_radius_scale,
        pair_scope=pair_scope,
        include_overlaps=include_overlaps,
        overlap_tolerance=overlap_tolerance,
    )


def has_too_close_atoms(
        mol: Any,
        *,
        minimum_distance: float = 0.50,
        covalent_radius_scale: Optional[float] = None,
        pair_scope: PairScope = "all",
        include_overlaps: bool = True,
        overlap_tolerance: float = 1.0e-3,
) -> bool:
    """Return whether an atom pair violates the requested separation."""
    return any(_iter_too_close_issues(
        _pair_table(mol),
        minimum_distance=minimum_distance,
        covalent_radius_scale=covalent_radius_scale,
        pair_scope=pair_scope,
        include_overlaps=include_overlaps,
        overlap_tolerance=overlap_tolerance,
    ))


def _segment_intersects_triangle(
        start: np.ndarray,
        end: np.ndarray,
        first: np.ndarray,
        second: np.ndarray,
        third: np.ndarray,
        tolerance: float,
        *,
        segment: bool = True,
) -> bool:
    """Moller-Trumbore intersection with the interior of one triangle."""
    direction = end - start
    edge1 = second - first
    edge2 = third - first
    cross = np.cross(direction, edge2)
    determinant = float(np.dot(edge1, cross))
    if abs(determinant) <= tolerance:
        return False

    inverse = 1.0 / determinant
    offset = start - first
    u = inverse * float(np.dot(offset, cross))
    if u < -tolerance or u > 1.0 + tolerance:
        return False

    offset_cross = np.cross(offset, edge1)
    v = inverse * float(np.dot(direction, offset_cross))
    if v < -tolerance or u + v > 1.0 + tolerance:
        return False

    parameter = inverse * float(np.dot(edge2, offset_cross))
    return not segment or tolerance < parameter < 1.0 - tolerance


def _planar_polygon_normal(
    points: np.ndarray,
    tolerance: float,
) -> Optional[np.ndarray]:
    """Return a unit normal when all polygon vertices share one plane."""
    normal = np.sum(
        np.cross(points, np.roll(points, -1, axis=0)),
        axis=0,
    )
    length = float(np.linalg.norm(normal))
    if length <= tolerance:
        return None

    normal /= length
    scale = max(1.0, float(np.ptp(points, axis=0).max()))
    distances = np.abs((points - points[0]) @ normal)
    return normal if np.all(distances <= tolerance * scale) else None


def _point_on_segment_2d(
    point: np.ndarray,
    start: np.ndarray,
    end: np.ndarray,
    tolerance: float,
) -> bool:
    edge = end - start
    offset = point - start
    scale = max(1.0, float(np.linalg.norm(edge)))
    cross = float(edge[0] * offset[1] - edge[1] * offset[0])
    if abs(cross) > tolerance * scale:
        return False
    projection = float(np.dot(offset, edge))
    return -tolerance <= projection <= float(np.dot(edge, edge)) + tolerance


def _point_in_polygon_2d(
    point: np.ndarray,
    polygon: np.ndarray,
    tolerance: float,
) -> bool:
    """Return whether a point lies in or on a simple, possibly concave polygon."""
    inside = False
    for index, start in enumerate(polygon):
        end = polygon[(index + 1) % len(polygon)]
        if _point_on_segment_2d(point, start, end, tolerance):
            return True
        if (start[1] > point[1]) != (end[1] > point[1]):
            crossing_x = start[0] + (
                (point[1] - start[1])
                * (end[0] - start[0])
                / (end[1] - start[1])
            )
            if point[0] < crossing_x:
                inside = not inside
    return inside


def _line_intersects_planar_polygon(
    points: np.ndarray,
    start: np.ndarray,
    end: np.ndarray,
    normal: np.ndarray,
    tolerance: float,
    *,
    segment: bool,
) -> bool:
    direction = end - start
    denominator = float(np.dot(normal, direction))
    if abs(denominator) <= tolerance * np.linalg.norm(direction):
        return False

    parameter = float(np.dot(normal, points[0] - start) / denominator)
    if segment and not tolerance < parameter < 1.0 - tolerance:
        return False

    intersection = start + parameter * direction
    projection_axes = np.delete(np.arange(3), np.argmax(np.abs(normal)))
    return _point_in_polygon_2d(
        intersection[projection_axes],
        points[:, projection_axes],
        tolerance,
    )


def _line_intersects_polygon(
        points: np.ndarray,
        start: np.ndarray,
        end: np.ndarray,
        *,
        tolerance: float,
        segment: bool,
) -> bool:
    """Return whether a line or segment intersects a polygonal ring surface."""
    if len(points) < 3 or not np.all(np.isfinite(points)):
        return False
    if not np.all(np.isfinite((start, end))):
        return False
    if np.linalg.norm(end - start) <= tolerance:
        return False

    normal = _planar_polygon_normal(points, tolerance)
    if normal is not None:
        return _line_intersects_planar_polygon(
            points,
            start,
            end,
            normal,
            tolerance,
            segment=segment,
        )

    # Preserve the historical center-fan surface for non-planar rings.
    center = np.mean(points, axis=0)
    return any(
        _segment_intersects_triangle(
            start,
            end,
            center,
            points[index],
            points[(index + 1) % len(points)],
            tolerance,
            segment=segment,
        )
        for index in range(len(points))
    )


def bond_intersects_ring(
        ring: Any,
        bond: Any,
        *,
        tolerance: float = 1.0e-8,
) -> bool:
    """Return whether a finite bond segment passes through a ring surface."""
    ring_atoms = tuple(ring.atoms)
    bond_atoms = (bond.atom1, bond.atom2)
    ring_atom_ids = {id(atom) for atom in ring_atoms}
    if any(id(atom) in ring_atom_ids for atom in bond_atoms):
        return False

    return _line_intersects_polygon(
        _atom_coordinates(ring_atoms),
        np.asarray(bond.atom1.coordinates, dtype=float),
        np.asarray(bond.atom2.coordinates, dtype=float),
        tolerance=tolerance,
        segment=True,
    )


def _rings_for_scope(mol: Any, ring_scope: RingScope) -> Sequence[Any]:
    if ring_scope not in ("full_graph", "ligand_skeleton"):
        raise ValueError(f"Unknown ring scope: {ring_scope!r}")
    uncached_rings = getattr(mol, "_uncached_rings", None)
    if uncached_rings is not None:
        return uncached_rings(ligand_skeleton=ring_scope == "ligand_skeleton")
    return mol.rings if ring_scope == "full_graph" else mol.ligand_rings


def _ring_key(ring: Any) -> Tuple[int, ...]:
    return tuple(sorted(_atom_index(atom, i) for i, atom in enumerate(ring.atoms)))


def _bond_key(bond: Any) -> Tuple[int, int]:
    first = _atom_index(bond.atom1, 0)
    second = _atom_index(bond.atom2, 1)
    return tuple(sorted((first, second)))


def find_bond_ring_intersections(
        mol: Any,
        *,
        ring_scope: RingScope = "full_graph",
        max_ring_size: int = 8,
) -> Tuple[Tuple[Any, Any], ...]:
    """Return stable ``(ring, bond)`` pairs for bonds crossing ring surfaces."""
    return tuple(_iter_bond_ring_intersections(
        mol,
        ring_scope=ring_scope,
        max_ring_size=max_ring_size,
    ))


def bond_ring_intersection_checks(
        mol: Any,
        intersections: Sequence[Tuple[Any, Any]],
) -> Tuple[GeometryCheck, ...]:
    """Convert detected crossings into stable, serializable quality checks."""
    if not intersections:
        return (GeometryCheck(
            name="bond_ring_intersection",
            passed=True,
            measured=0,
            threshold=0,
        ),)

    bond_positions = {
        _bond_key(candidate): index
        for index, candidate in enumerate(mol.bonds)
    }
    return tuple(
        GeometryCheck(
            name="bond_ring_intersection",
            passed=False,
            measured=_ring_key(ring),
            threshold=False,
            atom_indices=_bond_key(bond),
            bond_indices=(bond_positions[_bond_key(bond)],),
            message="A bond passes through a selected ring surface",
        )
        for ring, bond in intersections
    )


def _iter_bond_ring_intersections(
    mol: Any,
    *,
    ring_scope: RingScope,
    max_ring_size: int,
):
    rings = sorted(
        (
            ring for ring in _rings_for_scope(mol, ring_scope)
            if 3 <= len(ring) <= max_ring_size
        ),
        key=_ring_key,
    )
    bonds = sorted(mol.bonds, key=_bond_key)
    for ring in rings:
        for bond in bonds:
            if bond_intersects_ring(ring, bond):
                yield ring, bond


def has_bond_ring_intersection(
        mol: Any,
        *,
        ring_scope: RingScope = "full_graph",
        max_ring_size: int = 8,
) -> bool:
    """Return whether a bond crosses a ring in the selected topology view."""
    return any(_iter_bond_ring_intersections(
        mol,
        ring_scope=ring_scope,
        max_ring_size=max_ring_size,
    ))


def _point_segment_distance(
        point: np.ndarray,
        start: np.ndarray,
        end: np.ndarray,
) -> float:
    direction = end - start
    length_squared = float(np.dot(direction, direction))
    if length_squared == 0.0:
        return float(np.linalg.norm(point - start))
    parameter = float(np.dot(point - start, direction) / length_squared)
    closest = start + np.clip(parameter, 0.0, 1.0) * direction
    return float(np.linalg.norm(point - closest))


def _segment_distance(
        first_start: np.ndarray,
        first_end: np.ndarray,
        second_start: np.ndarray,
        second_end: np.ndarray,
        tolerance: float = 1.0e-12,
) -> float:
    """Return the shortest Euclidean distance between two finite segments."""
    first_direction = first_end - first_start
    second_direction = second_end - second_start
    offset = first_start - second_start
    first_length = float(np.dot(first_direction, first_direction))
    second_length = float(np.dot(second_direction, second_direction))

    if first_length <= tolerance and second_length <= tolerance:
        return float(np.linalg.norm(first_start - second_start))
    if first_length <= tolerance:
        return _point_segment_distance(first_start, second_start, second_end)
    if second_length <= tolerance:
        return _point_segment_distance(second_start, first_start, first_end)

    cross = float(np.dot(first_direction, second_direction))
    first_offset = float(np.dot(first_direction, offset))
    second_offset = float(np.dot(second_direction, offset))
    denominator = first_length * second_length - cross * cross
    first_numerator = 0.0
    first_denominator = denominator
    second_numerator = 0.0
    second_denominator = denominator

    if denominator <= tolerance:
        first_numerator = 0.0
        first_denominator = 1.0
        second_numerator = second_offset
        second_denominator = second_length
    else:
        first_numerator = cross * second_offset - first_offset * second_length
        second_numerator = first_length * second_offset - cross * first_offset
        if first_numerator < 0.0:
            first_numerator = 0.0
            second_numerator = second_offset
            second_denominator = second_length
        elif first_numerator > first_denominator:
            first_numerator = first_denominator
            second_numerator = second_offset + cross
            second_denominator = second_length

    if second_numerator < 0.0:
        second_numerator = 0.0
        if -first_offset < 0.0:
            first_numerator = 0.0
        elif -first_offset > first_length:
            first_numerator = first_denominator
        else:
            first_numerator = -first_offset
            first_denominator = first_length
    elif second_numerator > second_denominator:
        second_numerator = second_denominator
        if -first_offset + cross < 0.0:
            first_numerator = 0.0
        elif -first_offset + cross > first_length:
            first_numerator = first_denominator
        else:
            first_numerator = -first_offset + cross
            first_denominator = first_length

    first_parameter = (
        0.0 if abs(first_numerator) <= tolerance
        else first_numerator / first_denominator
    )
    second_parameter = (
        0.0 if abs(second_numerator) <= tolerance
        else second_numerator / second_denominator
    )
    separation = (
        offset
        + first_parameter * first_direction
        - second_parameter * second_direction
    )
    return float(np.linalg.norm(separation))


def closest_ring_edge_to_bond(ring: Any, bond: Any) -> Any:
    """Return the ring edge closest to a bond using finite-segment distance."""
    if ring.mol is not bond.mol or bond not in ring.mol.bonds:
        raise ValueError("The ring and bond must belong to the same molecule")

    start = np.asarray(bond.atom1.coordinates, dtype=float)
    end = np.asarray(bond.atom2.coordinates, dtype=float)

    def edge_distance(edge: Any) -> Tuple[float, Tuple[int, int]]:
        distance = _segment_distance(
            np.asarray(edge.atom1.coordinates, dtype=float),
            np.asarray(edge.atom2.coordinates, dtype=float),
            start,
            end,
        )
        return distance, _bond_key(edge)

    return min(ring.bonds, key=edge_distance)


def closest_ring_opening_edge(
        mol: Any,
        ring: Any,
        bond: Any,
        *,
        ring_scope: RingScope = "ligand_skeleton",
) -> Optional[Any]:
    """Return the closest single, non-fused ring edge suitable for temporary opening."""
    memberships = {}
    for candidate_ring in _rings_for_scope(mol, ring_scope):
        for edge in candidate_ring.bonds:
            key = _bond_key(edge)
            memberships[key] = memberships.get(key, 0) + 1

    eligible_edges = tuple(
        edge
        for edge in ring.bonds
        if float(edge.bond_order) == 1.0
        and memberships.get(_bond_key(edge), 0) == 1
    )
    if not eligible_edges:
        return None

    start = np.asarray(bond.atom1.coordinates, dtype=float)
    end = np.asarray(bond.atom2.coordinates, dtype=float)

    def edge_distance(edge: Any) -> Tuple[float, Tuple[int, int]]:
        distance = _segment_distance(
            np.asarray(edge.atom1.coordinates, dtype=float),
            np.asarray(edge.atom2.coordinates, dtype=float),
            start,
            end,
        )
        return distance, _bond_key(edge)

    return min(eligible_edges, key=edge_distance)


def _bond_kind(bond: Any) -> str:
    kind = getattr(bond, "bond_kind", "")
    return str(getattr(kind, "value", kind))


def _topology_bond_signature(
        bond: Any,
        positions: Mapping[int, int],
) -> BondTopologySignature:
    endpoints = tuple(sorted((
        positions[id(bond.atom1)],
        positions[id(bond.atom2)],
    )))
    return BondTopologySignature(
        atom_indices=endpoints,
        bond_order=float(bond.bond_order),
        bond_kind=_bond_kind(bond),
    )


def capture_topology(
    mol: Any,
    *,
    allow_added_hydrogens: bool = True,
) -> TopologyReference:
    """Capture the topology and whether preparation may append hydrogens."""
    atoms = tuple(mol.atoms)
    positions = {id(atom): i for i, atom in enumerate(atoms)}
    atom_signatures = tuple(
        AtomTopologySignature(
            index=i,
            atom_id=int(atom.id),
            atomic_number=int(atom.atomic_number),
            formal_charge=int(atom.formal_charge),
        )
        for i, atom in enumerate(atoms)
    )
    bond_signatures = tuple(sorted(
        (_topology_bond_signature(bond, positions) for bond in mol.bonds),
        key=lambda signature: signature.atom_indices,
    ))
    return TopologyReference(
        atom_signatures,
        bond_signatures,
        allow_added_hydrogens=allow_added_hydrogens,
    )


def _topology_checks(
        mol: Any,
        reference: TopologyReference,
) -> Tuple[GeometryCheck, ...]:
    atoms = tuple(mol.atoms)
    checks = []
    original_count = len(reference.atoms)

    if len(atoms) < original_count:
        checks.append(GeometryCheck(
            name="topology_atom_count",
            passed=False,
            measured=len(atoms),
            threshold=f">={original_count}",
            message="Original atoms were removed",
        ))
        return tuple(checks)

    for signature, atom in zip(reference.atoms, atoms[:original_count]):
        measured = (
            int(atom.id),
            int(atom.atomic_number),
            int(atom.formal_charge),
        )
        expected = (
            signature.atom_id,
            signature.atomic_number,
            signature.formal_charge,
        )
        if measured != expected:
            checks.append(GeometryCheck(
                name="topology_atom_identity",
                passed=False,
                measured=measured,
                threshold=expected,
                atom_indices=(signature.index,),
                message="An original atom identity or formal charge changed",
            ))

    added_indices = set(range(original_count, len(atoms)))
    if added_indices and not reference.allow_added_hydrogens:
        checks.append(GeometryCheck(
            name="topology_added_atoms",
            passed=False,
            measured=len(added_indices),
            threshold=0,
            atom_indices=tuple(sorted(added_indices)),
            message="Additional atoms are not allowed by this topology reference",
        ))
    elif added_indices:
        non_hydrogens = tuple(
            i for i in added_indices if int(atoms[i].atomic_number) != 1
        )
        if non_hydrogens:
            checks.append(GeometryCheck(
                name="topology_added_atoms",
                passed=False,
                measured=tuple(int(atoms[i].atomic_number) for i in non_hydrogens),
                threshold="hydrogen only",
                atom_indices=non_hydrogens,
                message="Only hydrogen atoms may be added during preparation",
            ))

    positions = {id(atom): i for i, atom in enumerate(atoms)}
    candidate_bonds = {
        signature.atom_indices: signature
        for signature in (
            _topology_bond_signature(bond, positions) for bond in mol.bonds
        )
    }
    reference_bonds = {
        signature.atom_indices: signature for signature in reference.bonds
    }

    for endpoints, expected in reference_bonds.items():
        measured = candidate_bonds.get(endpoints)
        if measured != expected:
            checks.append(GeometryCheck(
                name="topology_original_bond",
                passed=False,
                measured=measured,
                threshold=expected,
                atom_indices=endpoints,
                message="An original bond was removed or changed",
            ))

    added_bonds = set(candidate_bonds).difference(reference_bonds)
    invalid_added_bonds = tuple(sorted(
        endpoints
        for endpoints in added_bonds
        if not reference.allow_added_hydrogens
        or sum(endpoint in added_indices for endpoint in endpoints) != 1
    ))
    for endpoints in invalid_added_bonds:
        checks.append(GeometryCheck(
            name="topology_added_bond",
            passed=False,
            measured=endpoints,
            threshold="one added H endpoint",
            atom_indices=endpoints,
            message="Only new X-H bonds may be added during preparation",
        ))

    if reference.allow_added_hydrogens:
        degree = {index: 0 for index in added_indices}
        for endpoints in added_bonds:
            for endpoint in endpoints:
                if endpoint in degree:
                    degree[endpoint] += 1
        invalid_hydrogens = tuple(
            index for index, count in sorted(degree.items()) if count != 1
        )
        if invalid_hydrogens:
            checks.append(GeometryCheck(
                name="topology_added_hydrogen_degree",
                passed=False,
                measured=tuple(degree[index] for index in invalid_hydrogens),
                threshold=1,
                atom_indices=invalid_hydrogens,
                message="Each added hydrogen must have exactly one new bond",
            ))

    if not checks:
        checks.append(GeometryCheck(
            name="topology",
            passed=True,
            measured=(len(atoms), len(candidate_bonds)),
            threshold=(original_count, len(reference_bonds)),
            message="Original topology is preserved",
        ))
    return tuple(checks)


def _resolve_thresholds(
        thresholds: Optional[Union[GeometryQualityThresholds, Mapping[str, Any]]],
) -> GeometryQualityThresholds:
    if thresholds is None:
        return GeometryQualityThresholds()
    if isinstance(thresholds, GeometryQualityThresholds):
        return thresholds
    return replace(GeometryQualityThresholds(), **dict(thresholds))


def _report_value(report: Any, name: str) -> Any:
    if isinstance(report, Mapping):
        return report.get(name)
    return getattr(report, name, None)


def _forcefield_checks(
        report: Any,
        level: QualityLevel,
        thresholds: GeometryQualityThresholds,
        stage: ForceFieldStage,
) -> Tuple[GeometryCheck, ...]:
    if report is None:
        if level == "strict":
            return (GeometryCheck(
                name="forcefield_report",
                passed=False,
                measured=None,
                threshold="complete force-field report",
                message="Strict geometry validation requires force-field diagnostics",
            ),)
        return ()

    checks = []
    setup_succeeded = _report_value(report, "setup_succeeded")
    checks.append(GeometryCheck(
        name="forcefield_setup",
        passed=setup_succeeded is not None and bool(setup_succeeded),
        measured=setup_succeeded,
        threshold=True,
        message="Force-field setup must succeed",
    ))

    required_finite_fields = ["final_energy"]
    if stage == "final":
        required_finite_fields.extend(("rms_gradient", "max_gradient"))
    for field_name in required_finite_fields:
        value = _report_value(report, field_name)
        checks.append(GeometryCheck(
            name=f"finite_{field_name}",
            passed=value is not None and bool(np.isfinite(value)),
            measured=None if value is None else float(value),
            threshold="finite",
            message=f"{field_name.replace('_', ' ')} must be finite",
        ))

    if level in ("basic", "standard", "strict"):
        exploded = _report_value(report, "exploded")
        if exploded is not None or level == "strict":
            checks.append(GeometryCheck(
                name="backend_explosion",
                passed=exploded is not None and not bool(exploded),
                measured=exploded,
                threshold=False,
                message="The force-field backend detected an exploded structure",
            ))

    if stage == "final" and level in ("standard", "strict"):
        converged = _report_value(report, "converged")
        if converged is not None or level == "strict":
            checks.append(GeometryCheck(
                name="forcefield_convergence",
                passed=converged is not None and bool(converged),
                severity="error" if level == "strict" else "warning",
                measured=converged,
                threshold=True,
                message="The force-field backend did not report convergence",
            ))

    if stage == "final" and level == "strict":
        gradient_limits = (
            ("rms_gradient", thresholds.strict_rms_gradient),
            ("max_gradient", thresholds.strict_max_gradient),
        )
        for field_name, limit in gradient_limits:
            value = _report_value(report, field_name)
            if value is not None and np.isfinite(value):
                checks.append(GeometryCheck(
                    name=field_name,
                    passed=float(value) <= limit,
                    measured=float(value),
                    threshold=limit,
                    message=f"{field_name.replace('_', ' ')} exceeds the strict limit",
                ))

        segment_epochs_completed = _report_value(
            report,
            "segment_epochs_completed",
        )
        converged = bool(_report_value(report, "converged"))
        no_history_required = (
            segment_epochs_completed is not None
            and int(segment_epochs_completed) == 1
            and converged
        )
        stability_checks = (
            ("energy_changes", thresholds.strict_energy_change),
            ("max_displacements", thresholds.strict_max_displacement),
        )
        stability_observations = []
        for field_name, limit in stability_checks:
            history = _report_value(report, field_name)
            values = () if history is None else tuple(history)
            recent = values[-thresholds.strict_stability_window:]
            value = max(recent) if recent else None
            stability_observations.append(len(recent))
            checks.append(GeometryCheck(
                name=field_name.removesuffix("s"),
                passed=no_history_required or (
                    value is not None
                    and np.isfinite(value)
                    and float(value) <= limit
                ),
                measured=None if value is None else float(value),
                threshold=limit,
                message=f"{field_name.replace('_', ' ')} do not satisfy the strict limit",
            ))

        observations = min(stability_observations)
        if segment_epochs_completed is None:
            epochs_completed = _report_value(report, "epochs_completed")
            required_observations = (
                min(thresholds.strict_stability_window, int(epochs_completed))
                if epochs_completed is not None
                else thresholds.strict_stability_window
            )
        else:
            required_observations = min(
                thresholds.strict_stability_window,
                max(int(segment_epochs_completed) - 1, 0),
            )
        checks.append(GeometryCheck(
            name="stability_observations",
            passed=(
                no_history_required
                or required_observations > 0
                and observations >= required_observations
            ),
            measured=observations,
            threshold=required_observations,
            message="Strict validation requires a stable multi-epoch history",
        ))
    return tuple(checks)


def _bond_position_data(mol: Any, atoms: Sequence[Any]):
    positions = {id(atom): i for i, atom in enumerate(atoms)}
    for bond_index, bond in enumerate(mol.bonds):
        first = positions[id(bond.atom1)]
        second = positions[id(bond.atom2)]
        yield bond_index, bond, first, second


def _coordination_metrics(
        mol: Any,
        atoms: Sequence[Any],
        coordinates: np.ndarray,
) -> Tuple[dict, ...]:
    positions = {id(atom): i for i, atom in enumerate(atoms)}
    donors = {i: [] for i, atom in enumerate(atoms) if atom.is_metal}
    for bond in mol.bonds:
        if not bond.is_metal_ligand_bond:
            continue
        first = positions[id(bond.atom1)]
        second = positions[id(bond.atom2)]
        metal, donor = (
            (first, second) if atoms[first].is_metal else (second, first)
        )
        donors[metal].append(donor)

    environments = []
    for metal, donor_indices in sorted(donors.items()):
        donor_indices = sorted(donor_indices)
        vectors = [coordinates[index] - coordinates[metal] for index in donor_indices]
        distances = [float(np.linalg.norm(vector)) for vector in vectors]
        angles = []
        for first, second in combinations(vectors, 2):
            denominator = np.linalg.norm(first) * np.linalg.norm(second)
            if denominator > 0.0:
                cosine = np.clip(np.dot(first, second) / denominator, -1.0, 1.0)
                angles.append(float(np.degrees(np.arccos(cosine))))
        environments.append({
            "metal_index": _atom_index(atoms[metal], metal),
            "coordination_number": len(donor_indices),
            "donor_indices": tuple(
                _atom_index(atoms[index], index) for index in donor_indices
            ),
            "distances": tuple(distances),
            "angles": tuple(angles),
        })
    return tuple(environments)


def evaluate_geometry_quality(
        mol: Any,
        *,
        level: QualityLevel = "standard",
        topology_reference: Optional[TopologyReference] = None,
        forcefield_report: Any = None,
        forcefield_stage: ForceFieldStage = "final",
        thresholds: Optional[Union[GeometryQualityThresholds, Mapping[str, Any]]] = None,
) -> GeometryQualityReport:
    """Evaluate coordinate, topology, and force-field result integrity."""
    if level not in ("off", "basic", "standard", "strict"):
        raise ValueError(f"Unknown geometry quality level: {level!r}")
    if forcefield_stage not in ("candidate", "final"):
        raise ValueError(f"Unknown force-field stage: {forcefield_stage!r}")

    limits = _resolve_thresholds(thresholds)
    atoms = tuple(mol.atoms)
    coordinates = np.asarray(mol.coordinates, dtype=float)
    checks = []
    metrics = {
        "atom_count": len(atoms),
        "bond_count": len(mol.bonds),
    }

    expected_shape = (len(atoms), 3)
    shape_ok = coordinates.shape == expected_shape
    checks.append(GeometryCheck(
        name="coordinate_shape",
        passed=shape_ok,
        measured=tuple(coordinates.shape),
        threshold=expected_shape,
        message="Coordinates must contain one Cartesian row per atom",
    ))

    finite_ok = shape_ok and bool(np.all(np.isfinite(coordinates)))
    nonfinite_indices = ()
    if shape_ok and not finite_ok:
        nonfinite_indices = tuple(
            _atom_index(atoms[i], i)
            for i in np.flatnonzero(~np.all(np.isfinite(coordinates), axis=1))
        )
    checks.append(GeometryCheck(
        name="finite_coordinates",
        passed=finite_ok,
        measured=finite_ok,
        threshold=True,
        atom_indices=nonfinite_indices,
        message="All Cartesian coordinates must be finite",
    ))

    if topology_reference is not None:
        checks.extend(_topology_checks(mol, topology_reference))
    checks.extend(
        _forcefield_checks(forcefield_report, level, limits, forcefield_stage)
    )

    if not finite_ok:
        passed = all(check.passed or check.severity != "error" for check in checks)
        return GeometryQualityReport(level, passed, tuple(checks), metrics)

    table = _pair_table(mol, coordinates)
    if len(table.distances):
        metrics["minimum_pair_distance"] = float(np.min(table.distances))

    if level == "off":
        passed = all(check.passed or check.severity != "error" for check in checks)
        return GeometryQualityReport(level, passed, tuple(checks), metrics)

    overlaps = _overlap_issues(table, limits.overlap_tolerance)
    if overlaps:
        checks.extend(GeometryCheck(
            name="atom_overlap",
            passed=False,
            measured=issue.distance,
            threshold=issue.threshold,
            atom_indices=issue.atom_indices,
            message="Two atoms occupy indistinguishable coordinates",
        ) for issue in overlaps)
    else:
        checks.append(GeometryCheck(
            name="atom_overlap",
            passed=True,
            measured=0,
            threshold=limits.overlap_tolerance,
        ))

    basic_close_pairs = _too_close_issues(
        table,
        minimum_distance=limits.basic_minimum_distance,
        covalent_radius_scale=None,
        pair_scope="all",
        include_overlaps=False,
        overlap_tolerance=limits.overlap_tolerance,
    )
    close_pairs_by_atoms = {
        issue.atom_indices: issue for issue in basic_close_pairs
    }
    if level in ("standard", "strict"):
        standard_close_pairs = _too_close_issues(
            table,
            minimum_distance=limits.standard_minimum_distance,
            covalent_radius_scale=limits.standard_covalent_radius_scale,
            pair_scope="nonbonded",
            include_overlaps=False,
            overlap_tolerance=limits.overlap_tolerance,
        )
        close_pairs_by_atoms.update(
            (issue.atom_indices, issue) for issue in standard_close_pairs
        )
    close_pairs = tuple(
        close_pairs_by_atoms[key] for key in sorted(close_pairs_by_atoms)
    )
    if close_pairs:
        checks.extend(GeometryCheck(
            name="atom_too_close",
            passed=False,
            measured=issue.distance,
            threshold=issue.threshold,
            atom_indices=issue.atom_indices,
            message="An atom pair is closer than the allowed separation",
        ) for issue in close_pairs)
    else:
        checks.append(GeometryCheck(
            name="atom_too_close",
            passed=True,
            measured=0,
            threshold=(
                limits.basic_minimum_distance
                if level == "basic"
                else (
                    limits.basic_minimum_distance,
                    limits.standard_minimum_distance,
                    limits.standard_covalent_radius_scale,
                )
            ),
        ))

    maximum_bond_length = 0.0
    for bond_index, bond, first, second in _bond_position_data(mol, atoms):
        distance = float(np.linalg.norm(coordinates[first] - coordinates[second]))
        maximum_bond_length = max(maximum_bond_length, distance)
        atom_indices = (table.atom_indices[first], table.atom_indices[second])
        valid_length = 0.0 < distance <= limits.maximum_bond_distance
        if not valid_length:
            checks.append(GeometryCheck(
                name="bond_distance",
                passed=False,
                measured=distance,
                threshold=(0.0, limits.maximum_bond_distance),
                atom_indices=atom_indices,
                bond_indices=(bond_index,),
                message="An explicit bond has an invalid or exploded length",
            ))

        if level in ("standard", "strict"):
            radius_sum = (
                float(atoms[first].covalent_radius)
                + float(atoms[second].covalent_radius)
            )
            if radius_sum > 0.0:
                ratio = distance / radius_sum
                ratio_limits = (
                    limits.metal_ligand_bond_ratio
                    if bond.is_metal_ligand_bond
                    else limits.covalent_bond_ratio
                )
                if not ratio_limits[0] <= ratio <= ratio_limits[1]:
                    checks.append(GeometryCheck(
                        name="bond_length_ratio",
                        passed=False,
                        measured=ratio,
                        threshold=ratio_limits,
                        atom_indices=atom_indices,
                        bond_indices=(bond_index,),
                        message="Bond length is inconsistent with covalent radii",
                    ))
    metrics["maximum_bond_length"] = maximum_bond_length
    if not any(check.name == "bond_distance" for check in checks):
        checks.append(GeometryCheck(
            name="bond_distance",
            passed=True,
            measured=maximum_bond_length,
            threshold=(0.0, limits.maximum_bond_distance),
        ))
    if level in ("standard", "strict") and not any(
            check.name == "bond_length_ratio" for check in checks
    ):
        checks.append(GeometryCheck(
            name="bond_length_ratio",
            passed=True,
            measured=None,
            threshold=(
                limits.covalent_bond_ratio,
                limits.metal_ligand_bond_ratio,
            ),
        ))

    if level in ("standard", "strict"):
        intersections = find_bond_ring_intersections(
            mol,
            ring_scope="ligand_skeleton",
        )
        metrics["bond_ring_intersection_count"] = len(intersections)
        checks.extend(bond_ring_intersection_checks(mol, intersections))
        metrics["coordination_environments"] = _coordination_metrics(
            mol,
            atoms,
            coordinates,
        )

    passed = all(check.passed or check.severity != "error" for check in checks)
    return GeometryQualityReport(level, passed, tuple(checks), metrics)


def is_geometry_reasonable(
        mol: Any,
        *,
        level: QualityLevel = "standard",
        topology_reference: Optional[TopologyReference] = None,
        forcefield_report: Any = None,
        forcefield_stage: ForceFieldStage = "final",
        thresholds: Optional[Union[GeometryQualityThresholds, Mapping[str, Any]]] = None,
) -> bool:
    """Return the pass/fail result of :func:`evaluate_geometry_quality`."""
    return evaluate_geometry_quality(
        mol,
        level=level,
        topology_reference=topology_reference,
        forcefield_report=forcefield_report,
        forcefield_stage=forcefield_stage,
        thresholds=thresholds,
    ).passed
