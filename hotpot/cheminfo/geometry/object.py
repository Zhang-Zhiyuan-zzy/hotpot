"""Immutable value objects for three-dimensional Euclidean geometry."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite, sqrt
from typing import Iterable, Iterator, Tuple, Union


__all__ = [
    "Point",
    "Line",
    "Segment",
    "Plane",
    "Triangle",
    "Cycle",
]


Coordinates3 = Tuple[float, float, float]
PointInput = Union["Point", Iterable[float]]


def _coordinates3(coordinates: Iterable[float]) -> Coordinates3:
    values = tuple(float(value) for value in coordinates)
    if len(values) != 3:
        raise ValueError("three-dimensional coordinates require exactly three values")
    return values[0], values[1], values[2]


def _point3(point: PointInput) -> "Point":
    if isinstance(point, Point):
        return point
    return Point.from_coordinates(point)


def _cycle_edges(vertices: Tuple["Point", ...]) -> Tuple["Segment", ...]:
    return tuple(
        Segment(vertices[index], vertices[(index + 1) % len(vertices)])
        for index in range(len(vertices))
    )


@dataclass(frozen=True, init=False)
class Point:
    """A point represented by three coordinates in one consistent length unit."""

    coordinates: Coordinates3

    def __init__(self, coordinates: Iterable[float]) -> None:
        object.__setattr__(self, "coordinates", _coordinates3(coordinates))

    @classmethod
    def from_coordinates(cls, coordinates: Iterable[float]) -> "Point":
        """Construct a point from an iterable of three coordinate values."""

        return cls(coordinates)

    @property
    def x(self) -> float:
        return self.coordinates[0]

    @property
    def y(self) -> float:
        return self.coordinates[1]

    @property
    def z(self) -> float:
        return self.coordinates[2]

    def __iter__(self) -> Iterator[float]:
        return iter(self.coordinates)


@dataclass(frozen=True, init=False)
class Line:
    """An infinite line represented by an origin and a direction vector."""

    origin: Point
    direction: Coordinates3

    def __init__(self, origin: PointInput, direction: Iterable[float]) -> None:
        object.__setattr__(self, "origin", _point3(origin))
        object.__setattr__(self, "direction", _coordinates3(direction))

    @classmethod
    def from_points(cls, first: PointInput, second: PointInput) -> "Line":
        """Construct an infinite line through two points."""

        first_point = _point3(first)
        second_point = _point3(second)
        direction = (
            second_point.x - first_point.x,
            second_point.y - first_point.y,
            second_point.z - first_point.z,
        )
        return cls(first_point, direction)


@dataclass(frozen=True, init=False)
class Segment:
    """A finite closed line segment between two points."""

    start: Point
    end: Point

    def __init__(self, start: PointInput, end: PointInput) -> None:
        object.__setattr__(self, "start", _point3(start))
        object.__setattr__(self, "end", _point3(end))

    @property
    def direction(self) -> Coordinates3:
        return (
            self.end.x - self.start.x,
            self.end.y - self.start.y,
            self.end.z - self.start.z,
        )

    @property
    def length(self) -> float:
        return sqrt(sum(component * component for component in self.direction))


@dataclass(frozen=True, init=False)
class Plane:
    """A plane represented by one point and a unit normal when finite."""

    point: Point
    normal: Coordinates3

    def __init__(self, point: PointInput, normal: Iterable[float]) -> None:
        normal_coordinates = _coordinates3(normal)
        normal_length = sqrt(sum(value * value for value in normal_coordinates))
        if normal_length == 0.0:
            raise ValueError("a plane normal cannot be the zero vector")
        if isfinite(normal_length):
            normal_coordinates = (
                normal_coordinates[0] / normal_length,
                normal_coordinates[1] / normal_length,
                normal_coordinates[2] / normal_length,
            )

        object.__setattr__(self, "point", _point3(point))
        object.__setattr__(self, "normal", normal_coordinates)


@dataclass(frozen=True, init=False)
class Triangle:
    """An ordered triangle whose vertices may be geometrically degenerate."""

    first: Point
    second: Point
    third: Point

    def __init__(
        self,
        first: PointInput,
        second: PointInput,
        third: PointInput,
    ) -> None:
        object.__setattr__(self, "first", _point3(first))
        object.__setattr__(self, "second", _point3(second))
        object.__setattr__(self, "third", _point3(third))

    @property
    def vertices(self) -> Tuple[Point, Point, Point]:
        return self.first, self.second, self.third

    @property
    def edges(self) -> Tuple[Segment, Segment, Segment]:
        return (
            Segment(self.first, self.second),
            Segment(self.second, self.third),
            Segment(self.third, self.first),
        )


@dataclass(frozen=True, init=False)
class Cycle:
    """An ordered closed boundary independent of chemical ring perception.

    A cycle stores only geometry.  It neither finds cycles in a graph nor
    records which ring-perception algorithm supplied its ordered vertices.
    """

    vertices: Tuple[Point, ...]

    def __init__(self, vertices: Iterable[PointInput]) -> None:
        vertex_tuple = tuple(_point3(vertex) for vertex in vertices)
        if len(vertex_tuple) < 3:
            raise ValueError("a cycle requires at least three vertices")
        object.__setattr__(self, "vertices", vertex_tuple)

    @property
    def edges(self) -> Tuple[Segment, ...]:
        return _cycle_edges(self.vertices)

    def __len__(self) -> int:
        return len(self.vertices)

    def __iter__(self) -> Iterator[Point]:
        return iter(self.vertices)
