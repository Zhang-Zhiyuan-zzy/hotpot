"""Numerical settings shared by geometry relation kernels."""

from __future__ import annotations

from dataclasses import dataclass, field
from math import isfinite


__all__ = [
    "NumericToleranceSettings",
    "SurfaceEnumerationSettings",
    "GeometrySettings",
    "DEFAULT_GEOMETRY_SETTINGS",
]


def _require_finite(name: str, value: float) -> None:
    if not isfinite(value):
        raise ValueError(f"{name} must be finite")


@dataclass(frozen=True)
class NumericToleranceSettings:
    """Dimension-aware tolerances used by geometry predicates."""

    absolute_length: float = 1.0e-8
    relative_length: float = 1.0e-10
    parameter: float = 1.0e-10
    machine_epsilon_factor: float = 64.0
    predicate_guard_factor: float = 4.0
    planarity_factor: float = 1.0
    winding_residual: float = 1.0e-10
    intersection_merge_factor: float = 4.0
    aabb_padding_factor: float = 4.0

    def __post_init__(self) -> None:
        for name in (
            "absolute_length",
            "relative_length",
            "parameter",
            "machine_epsilon_factor",
            "predicate_guard_factor",
            "planarity_factor",
            "winding_residual",
            "intersection_merge_factor",
            "aabb_padding_factor",
        ):
            _require_finite(name, getattr(self, name))

        if self.absolute_length <= 0.0:
            raise ValueError("absolute_length must be greater than zero")
        if self.relative_length < 0.0:
            raise ValueError("relative_length must be non-negative")
        if self.predicate_guard_factor <= 1.0:
            raise ValueError("predicate_guard_factor must be greater than one")
        if not 0.0 < self.parameter < 1.0 / (2.0 * self.predicate_guard_factor):
            raise ValueError(
                "parameter must be greater than zero and smaller than "
                "1 / (2 * predicate_guard_factor)"
            )
        if self.machine_epsilon_factor < 1.0:
            raise ValueError("machine_epsilon_factor must be at least one")
        if self.planarity_factor <= 0.0:
            raise ValueError("planarity_factor must be greater than zero")
        if not 0.0 < self.winding_residual < 0.5:
            raise ValueError("winding_residual must be between zero and one half")
        if self.intersection_merge_factor < 1.0:
            raise ValueError("intersection_merge_factor must be at least one")
        if self.aabb_padding_factor < self.predicate_guard_factor:
            raise ValueError(
                "aabb_padding_factor must be at least predicate_guard_factor"
            )


@dataclass(frozen=True)
class SurfaceEnumerationSettings:
    """Budgets for exhaustive cycle-surface enumeration."""

    maximum_cycle_vertices: int = 8
    maximum_surface_count: int = 132
    maximum_segment_triangle_tests: int = 792
    maximum_triangle_pair_tests: int = 1980

    def __post_init__(self) -> None:
        if self.maximum_cycle_vertices < 3:
            raise ValueError("maximum_cycle_vertices must be at least three")
        if self.maximum_surface_count <= 0:
            raise ValueError("maximum_surface_count must be greater than zero")
        if self.maximum_segment_triangle_tests <= 0:
            raise ValueError(
                "maximum_segment_triangle_tests must be greater than zero"
            )
        if self.maximum_triangle_pair_tests <= 0:
            raise ValueError("maximum_triangle_pair_tests must be greater than zero")


@dataclass(frozen=True)
class GeometrySettings:
    """Complete numerical configuration for geometry relations."""

    tolerance: NumericToleranceSettings = field(
        default_factory=NumericToleranceSettings
    )
    surface: SurfaceEnumerationSettings = field(
        default_factory=SurfaceEnumerationSettings
    )


DEFAULT_GEOMETRY_SETTINGS = GeometrySettings()
