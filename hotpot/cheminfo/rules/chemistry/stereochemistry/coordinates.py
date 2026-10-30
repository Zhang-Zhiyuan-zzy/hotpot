"""Pure stereochemistry geometry classifiers and coordinate diagnostics."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from math import (
    radians,
    sqrt,
)
import numpy as np
from numpy.typing import NDArray
from typing import Mapping
from ....structure.stereo import (
    StereoCarrierKind,
    StereoIssueCode,
    StereoPayload,
    StereoRecord,
    StereoValidationIssue,
    StereoValidationReport,
)
from .algebra import (
    _COORDINATION_CARDINALITIES,
    _COORDINATION_FAMILIES,
    _COORDINATION_REPRESENTATIVES,
    _Gate7CoordinationSite,
    _Level2Descriptor,
    _Level2DescriptorFamily,
    _coordinate_group_consistency_issues,
    _gate7_site_from_record,
    _parse_level2_descriptor,
    _record_atom_indices,
)


__all__ = ()

_GEOMETRY_RELATIVE_TOLERANCE = 1.0e-8


_COORDINATION_IDEAL_DIRECTIONS = {
    _Level2DescriptorFamily.SQUARE_PLANAR: (
        (1.0, 0.0, 0.0),
        (0.0, 1.0, 0.0),
        (-1.0, 0.0, 0.0),
        (0.0, -1.0, 0.0),
    ),
    _Level2DescriptorFamily.TRIGONAL_BIPYRAMIDAL: (
        (0.0, 0.0, 1.0),
        (0.0, 0.0, -1.0),
        (1.0, 0.0, 0.0),
        (-0.5, sqrt(3.0) / 2.0, 0.0),
        (-0.5, -sqrt(3.0) / 2.0, 0.0),
    ),
    _Level2DescriptorFamily.SQUARE_PYRAMIDAL: (
        (0.0, 0.0, 1.0),
        (1.0, 0.0, 0.0),
        (0.0, 1.0, 0.0),
        (-1.0, 0.0, 0.0),
        (0.0, -1.0, 0.0),
    ),
    _Level2DescriptorFamily.OCTAHEDRAL: (
        (1.0, 0.0, 0.0),
        (-1.0, 0.0, 0.0),
        (0.0, 1.0, 0.0),
        (0.0, -1.0, 0.0),
        (0.0, 0.0, 1.0),
        (0.0, 0.0, -1.0),
    ),
}


_GATE7_ACCEPTANCE_RADIANS = radians(12.0)


_GATE7_SAME_FAMILY_MARGIN_RADIANS = radians(2.0)


_GATE7_FIVE_COORDINATE_MARGIN_RADIANS = radians(3.0)


_GATE7_FIVE_COORDINATE_FAMILIES = (
    _Level2DescriptorFamily.TRIGONAL_BIPYRAMIDAL,
    _Level2DescriptorFamily.SQUARE_PYRAMIDAL,
)


@dataclass(frozen=True)
class _Gate7CoordinateCandidate:
    """One descriptor fit ranked by raw angular RMS in radians."""

    descriptor: _Level2Descriptor
    score_radians: float


def _geometry_issue(
    message: str,
    *,
    record_index: int | None = None,
    atom_indices: tuple[int, ...] = (),
) -> StereoValidationIssue:
    return StereoValidationIssue(
        code=StereoIssueCode.DEGENERATE_GEOMETRY,
        message=message,
        record_index=record_index,
        atom_indices=atom_indices,
    )


def _finite_coordinate_array(
    atom_count: int,
    coordinates: NDArray[np.float64],
) -> tuple[NDArray[np.float64] | None, StereoValidationIssue | None]:
    """Return a validated coordinate view with the frozen molecule row shape."""

    array = np.asarray(coordinates, dtype=np.float64)
    expected_shape = (atom_count, 3)
    if array.shape != expected_shape:
        return None, _geometry_issue(
            f"coordinate array must have shape {expected_shape}, got {array.shape}"
        )
    if not np.all(np.isfinite(array)):
        nonfinite_rows = tuple(
            int(row) for row in np.flatnonzero(~np.all(np.isfinite(array), axis=1))
        )
        return None, _geometry_issue(
            "coordinate array contains non-finite values",
            atom_indices=nonfinite_rows,
        )
    return array, None


def _normalized_geometry_sign(
    value: float,
    scale: float,
) -> int:
    """Classify a signed measure using a scale-free deterministic tolerance."""

    if scale == 0.0:
        return 0
    normalized = value / scale
    if abs(normalized) <= _GEOMETRY_RELATIVE_TOLERANCE:
        return 0
    return 1 if normalized > 0.0 else -1


def _is_relatively_degenerate(length: float, local_scale: float) -> bool:
    """Return whether a local length vanishes at the geometry tolerance."""

    return local_scale == 0.0 or length / local_scale <= _GEOMETRY_RELATIVE_TOLERANCE


def _gate7_ideal_carrier_directions(
    descriptor: _Level2Descriptor,
) -> NDArray[np.float64]:
    """Return one ideal configuration in stored carrier order."""

    ideal_sites = _COORDINATION_IDEAL_DIRECTIONS[descriptor.family]
    representative = _COORDINATION_REPRESENTATIVES[descriptor.family][
        int(descriptor.value) - 1
    ]
    directions = np.empty((len(representative), 3), dtype=np.float64)
    for site, carrier_position in enumerate(representative):
        directions[carrier_position] = ideal_sites[site]
    return directions


def _gate7_proper_rotation_fit(
    ideal: NDArray[np.float64],
    observed: NDArray[np.float64],
) -> tuple[NDArray[np.float64], float]:
    """Fit equal-weight unit directions by no-centroid proper Procrustes.

    Kabsch minimizes squared Euclidean chord lengths. The returned score is
    computed afterwards as angular RMS; it is not the optimized objective.
    """

    left, _, right_transpose = np.linalg.svd(ideal.T @ observed)
    correction = np.eye(3, dtype=np.float64)
    correction[-1, -1] = (
        1.0
        if np.linalg.det(left @ right_transpose) >= 0.0
        else -1.0
    )
    rotation = left @ correction @ right_transpose
    fitted = ideal @ rotation
    dots = np.clip(np.sum(fitted * observed, axis=1), -1.0, 1.0)
    cross_norms = np.linalg.norm(np.cross(fitted, observed), axis=1)
    angles = np.arctan2(cross_norms, dots)
    score = float(np.sqrt(np.mean(np.square(angles))))
    return rotation, score


def _gate7_coordination_candidates(
    family: _Level2DescriptorFamily,
    observed: NDArray[np.float64],
) -> tuple[_Gate7CoordinateCandidate, ...]:
    """Rank every descriptor in one coordination family deterministically."""

    candidates = []
    for value in range(1, _COORDINATION_CARDINALITIES[family] + 1):
        descriptor = _Level2Descriptor(family, value)
        ideal = _gate7_ideal_carrier_directions(descriptor)
        _, score = _gate7_proper_rotation_fit(ideal, observed)
        candidates.append(_Gate7CoordinateCandidate(descriptor, score))
    return tuple(
        sorted(
            candidates,
            key=lambda candidate: (
                candidate.score_radians,
                int(candidate.descriptor.value),
            ),
        )
    )


def _select_gate7_coordination_descriptor(
    declared_family: _Level2DescriptorFamily,
    candidates_by_family: Mapping[
        _Level2DescriptorFamily,
        Sequence[_Gate7CoordinateCandidate],
    ],
    *,
    record_index: int,
    atom_indices: tuple[int, ...],
) -> tuple[_Level2Descriptor | None, StereoValidationIssue | None]:
    """Apply the raw-radian 12/2/3-degree policy to ranked candidates."""

    declared = candidates_by_family[declared_family]
    declared_best = declared[0]

    if declared_family in _GATE7_FIVE_COORDINATE_FAMILIES:
        alternate_family = next(
            family
            for family in _GATE7_FIVE_COORDINATE_FAMILIES
            if family is not declared_family
        )
        alternate_best = candidates_by_family[alternate_family][0]
        if (
            declared_best.score_radians > _GATE7_ACCEPTANCE_RADIANS
            and alternate_best.score_radians > _GATE7_ACCEPTANCE_RADIANS
        ):
            return None, _geometry_issue(
                "TBP/SPY coordinate geometry exceeds the 12-degree angular "
                "RMS acceptance cutoff in both families",
                record_index=record_index,
                atom_indices=atom_indices,
            )
        family_gap = abs(
            declared_best.score_radians - alternate_best.score_radians
        )
        if family_gap <= _GATE7_FIVE_COORDINATE_MARGIN_RADIANS:
            return None, _geometry_issue(
                "TBP/SPY coordinate geometry is ambiguous within the "
                "3-degree family margin",
                record_index=record_index,
                atom_indices=atom_indices,
            )
        if alternate_best.score_radians < declared_best.score_radians:
            return None, StereoValidationIssue(
                code=StereoIssueCode.COORDINATE_CONTRADICTION,
                message=(
                    f"declared {declared_family.value} family contradicts "
                    f"coordinate-preferred {alternate_family.value} family"
                ),
                record_index=record_index,
                atom_indices=atom_indices,
            )
    elif declared_best.score_radians > _GATE7_ACCEPTANCE_RADIANS:
        return None, _geometry_issue(
            f"{declared_family.value} coordinate geometry exceeds the "
            "12-degree angular RMS acceptance cutoff",
            record_index=record_index,
            atom_indices=atom_indices,
        )

    if (
        declared[1].score_radians - declared_best.score_radians
        <= _GATE7_SAME_FAMILY_MARGIN_RADIANS
    ):
        return None, _geometry_issue(
            f"{declared_family.value} coordinate geometry is ambiguous within "
            "the 2-degree same-family margin",
            record_index=record_index,
            atom_indices=atom_indices,
        )
    return declared_best.descriptor, None


def _gate7_center_rooted_directions(
    site: _Gate7CoordinationSite,
    coordinates: NDArray[np.float64],
) -> tuple[NDArray[np.float64] | None, StereoValidationIssue | None]:
    """Normalize each center-to-carrier vector without donor centering."""

    rows = (site.center, *site.carrier_rows)
    site_coordinates = coordinates[np.asarray(rows)]
    coordinate_scale = float(np.max(np.abs(site_coordinates)))
    scaled_coordinates = np.zeros_like(site_coordinates, dtype=np.float64)
    if coordinate_scale != 0.0:
        scaled_coordinates = site_coordinates / coordinate_scale
    vectors = scaled_coordinates[1:] - scaled_coordinates[0]
    component_scales = np.max(np.abs(vectors), axis=1)
    scaled_vectors = np.zeros_like(vectors, dtype=np.float64)
    np.divide(
        vectors,
        component_scales[:, np.newaxis],
        out=scaled_vectors,
        where=component_scales[:, np.newaxis] != 0.0,
    )
    scaled_norms = np.linalg.norm(scaled_vectors, axis=1)
    max_component_scale = float(np.max(component_scales))
    relative_norms = np.zeros_like(component_scales, dtype=np.float64)
    if max_component_scale != 0.0:
        relative_norms = (
            component_scales / max_component_scale
        ) * scaled_norms
    local_scale = float(np.max(relative_norms))
    if any(
        _is_relatively_degenerate(float(norm), local_scale)
        for norm in relative_norms
    ):
        return None, _geometry_issue(
            f"{site.descriptor.family.value} coordinate geometry has a carrier "
            "coincident with its center within relative tolerance "
            f"{_GEOMETRY_RELATIVE_TOLERANCE:g}",
            record_index=site.record_index,
            atom_indices=(site.center, *site.carrier_rows),
        )
    directions = scaled_vectors / scaled_norms[:, np.newaxis]
    return directions, None


def _gate7_coordination_coordinate_descriptor(
    site: _Gate7CoordinationSite,
    coordinates: NDArray[np.float64],
) -> tuple[_Level2Descriptor | None, StereoValidationIssue | None]:
    """Assign one private Gate-7 descriptor without public cutover."""

    directions, issue = _gate7_center_rooted_directions(site, coordinates)
    if issue is not None:
        return None, issue

    family = site.descriptor.family
    families = [family]
    if family in _GATE7_FIVE_COORDINATE_FAMILIES:
        families.extend(
            candidate_family
            for candidate_family in _GATE7_FIVE_COORDINATE_FAMILIES
            if candidate_family is not family
        )
    candidates = {
        candidate_family: _gate7_coordination_candidates(
            candidate_family,
            directions,
        )
        for candidate_family in families
    }
    return _select_gate7_coordination_descriptor(
        family,
        candidates,
        record_index=site.record_index,
        atom_indices=(site.center, *site.carrier_rows),
    )


def _tetrahedral_coordinate_descriptor(
    record: StereoRecord,
    coordinates: NDArray[np.float64],
    *,
    record_index: int,
) -> tuple[_Level2Descriptor | None, StereoValidationIssue | None]:
    """Assign TH parity from the ordered-carrier signed tetrahedral volume.

    Positive normalized volume is ``even`` and negative volume is ``odd``.
    An implicit carrier uses the anchor as an orientation-only zero-vector
    proxy.  This preserves ordered-carrier parity without inventing a bond
    length or direction; three explicit bonds coplanar with the anchor remain
    correctly unresolved.
    """

    center = record.anchor.atom_indices[0]
    center_coordinate = coordinates[center]
    vectors = []
    explicit_vectors = []
    for carrier in record.carriers:
        if carrier.kind is StereoCarrierKind.ATOM:
            vector = coordinates[carrier.atom] - center_coordinate
            vectors.append(vector)
            explicit_vectors.append(vector)
        else:
            vectors.append(np.zeros(3, dtype=np.float64))

    explicit_norms = tuple(float(np.linalg.norm(vector)) for vector in explicit_vectors)
    local_scale = max(explicit_norms, default=0.0)
    if any(_is_relatively_degenerate(norm, local_scale) for norm in explicit_norms):
        return None, _geometry_issue(
            "TH coordinate geometry has a carrier coincident with its center "
            "within relative tolerance "
            f"{_GEOMETRY_RELATIVE_TOLERANCE:g}",
            record_index=record_index,
            atom_indices=_record_atom_indices(record),
        )

    carrier_coordinates = tuple(center_coordinate + vector for vector in vectors)
    characteristic_length = max(
        float(np.linalg.norm(carrier_coordinates[left] - carrier_coordinates[right]))
        for left in range(4)
        for right in range(left)
    )
    if characteristic_length == 0.0:
        sign = 0
    else:
        first = (
            carrier_coordinates[0] - carrier_coordinates[3]
        ) / characteristic_length
        second = (
            carrier_coordinates[1] - carrier_coordinates[3]
        ) / characteristic_length
        third = (
            carrier_coordinates[2] - carrier_coordinates[3]
        ) / characteristic_length
        signed_volume = float(np.dot(first, np.cross(second, third)))
        sign = _normalized_geometry_sign(signed_volume, 1.0)
    if sign == 0:
        return None, _geometry_issue(
            "TH signed volume is degenerate within relative tolerance "
            f"{_GEOMETRY_RELATIVE_TOLERANCE:g}",
            record_index=record_index,
            atom_indices=_record_atom_indices(record),
        )
    return (
        _Level2Descriptor(
            _Level2DescriptorFamily.TETRAHEDRAL,
            "even" if sign > 0 else "odd",
        ),
        None,
    )


def _double_bond_coordinate_descriptor(
    record: StereoRecord,
    coordinates: NDArray[np.float64],
    *,
    record_index: int,
) -> tuple[_Level2Descriptor | None, StereoValidationIssue | None]:
    """Assign DB relation from endpoint-scoped ordered transverse vectors."""

    begin, end = record.anchor.atom_indices
    axis = coordinates[end] - coordinates[begin]
    axis_norm = float(np.linalg.norm(axis))
    endpoint_vectors = tuple(
        tuple(
            coordinates[carrier.atom] - coordinates[endpoint]
            for carrier in record.carriers[2 * endpoint_index : 2 * endpoint_index + 2]
            if carrier.kind is StereoCarrierKind.ATOM
        )
        for endpoint_index, endpoint in enumerate((begin, end))
    )
    explicit_norms = tuple(
        float(np.linalg.norm(vector))
        for vectors in endpoint_vectors
        for vector in vectors
    )
    local_scale = max((axis_norm, *explicit_norms))
    if _is_relatively_degenerate(axis_norm, local_scale):
        return None, _geometry_issue(
            "DB coordinate geometry has coincident bond endpoints within "
            f"relative tolerance {_GEOMETRY_RELATIVE_TOLERANCE:g}",
            record_index=record_index,
            atom_indices=_record_atom_indices(record),
        )
    if any(_is_relatively_degenerate(norm, local_scale) for norm in explicit_norms):
        return None, _geometry_issue(
            "DB coordinate geometry has a carrier coincident with its endpoint "
            "within relative tolerance "
            f"{_GEOMETRY_RELATIVE_TOLERANCE:g}",
            record_index=record_index,
            atom_indices=_record_atom_indices(record),
        )
    axis_unit = axis / axis_norm

    endpoint_directions = []
    for endpoint_index, endpoint in enumerate((begin, end)):
        transverse = []
        group = record.carriers[2 * endpoint_index : 2 * endpoint_index + 2]
        for carrier in group:
            if carrier.kind is StereoCarrierKind.ATOM:
                vector = coordinates[carrier.atom] - coordinates[endpoint]
                transverse.append(vector - np.dot(vector, axis_unit) * axis_unit)
            else:
                transverse.append(np.zeros(3, dtype=np.float64))

        direction = transverse[0] - transverse[1]
        direction_norm = float(np.linalg.norm(direction))
        endpoint_scale = max(
            (
                axis_norm,
                *(
                    float(np.linalg.norm(vector))
                    for vector in endpoint_vectors[endpoint_index]
                ),
            )
        )
        if _is_relatively_degenerate(direction_norm, endpoint_scale):
            return None, _geometry_issue(
                f"DB endpoint {endpoint} has degenerate transverse carriers "
                "within relative tolerance "
                f"{_GEOMETRY_RELATIVE_TOLERANCE:g}",
                record_index=record_index,
                atom_indices=_record_atom_indices(record),
            )
        endpoint_directions.append(direction)

    relation = float(
        np.dot(
            endpoint_directions[0] / np.linalg.norm(endpoint_directions[0]),
            endpoint_directions[1] / np.linalg.norm(endpoint_directions[1]),
        )
    )
    sign = _normalized_geometry_sign(relation, 1.0)
    if sign == 0:
        return None, _geometry_issue(
            "DB transverse relation is degenerate within relative tolerance "
            f"{_GEOMETRY_RELATIVE_TOLERANCE:g}",
            record_index=record_index,
            atom_indices=_record_atom_indices(record),
        )
    return (
        _Level2Descriptor(
            _Level2DescriptorFamily.DOUBLE_BOND,
            "together" if sign > 0 else "opposite",
        ),
        None,
    )


def _axial_coordinate_descriptor(
    record: StereoRecord,
    coordinates: NDArray[np.float64],
    *,
    record_index: int,
) -> tuple[_Level2Descriptor | None, StereoValidationIssue | None]:
    """Assign AX parity from its ordered endpoint carrier-pair triple product."""

    begin = record.anchor.atom_indices[0]
    end = record.anchor.atom_indices[-1]
    axis = coordinates[end] - coordinates[begin]
    axis_norm = float(np.linalg.norm(axis))
    endpoint_vectors = tuple(
        tuple(
            coordinates[carrier.atom] - coordinates[endpoint]
            for carrier in record.carriers[
                2 * endpoint_index : 2 * endpoint_index + 2
            ]
            if carrier.kind is StereoCarrierKind.ATOM
        )
        for endpoint_index, endpoint in enumerate((begin, end))
    )
    explicit_norms = tuple(
        float(np.linalg.norm(vector))
        for vectors in endpoint_vectors
        for vector in vectors
    )
    local_scale = max((axis_norm, *explicit_norms))
    if _is_relatively_degenerate(axis_norm, local_scale):
        return None, _geometry_issue(
            "AX coordinate geometry has coincident axis endpoints within "
            f"relative tolerance {_GEOMETRY_RELATIVE_TOLERANCE:g}",
            record_index=record_index,
            atom_indices=_record_atom_indices(record),
        )
    axis_unit = axis / axis_norm

    endpoint_directions = []
    for endpoint_index, endpoint in enumerate((begin, end)):
        explicit_at_endpoint = endpoint_vectors[endpoint_index]
        endpoint_scale = max(
            (
                axis_norm,
                *(
                    float(np.linalg.norm(vector))
                    for vector in explicit_at_endpoint
                ),
            )
        )
        if any(
            _is_relatively_degenerate(
                float(np.linalg.norm(vector)),
                endpoint_scale,
            )
            for vector in explicit_at_endpoint
        ):
            return None, _geometry_issue(
                "AX coordinate geometry has a carrier coincident with its endpoint "
                "within relative tolerance "
                f"{_GEOMETRY_RELATIVE_TOLERANCE:g}",
                record_index=record_index,
                atom_indices=_record_atom_indices(record),
            )

        transverse = []
        group = record.carriers[2 * endpoint_index : 2 * endpoint_index + 2]
        for carrier in group:
            if carrier.kind is StereoCarrierKind.ATOM:
                vector = coordinates[carrier.atom] - coordinates[endpoint]
                transverse.append(vector - np.dot(vector, axis_unit) * axis_unit)
            else:
                transverse.append(np.zeros(3, dtype=np.float64))

        direction = transverse[0] - transverse[1]
        direction_norm = float(np.linalg.norm(direction))
        if _is_relatively_degenerate(direction_norm, endpoint_scale):
            return None, _geometry_issue(
                f"AX endpoint {endpoint} has degenerate projected carriers "
                "within relative tolerance "
                f"{_GEOMETRY_RELATIVE_TOLERANCE:g}",
                record_index=record_index,
                atom_indices=_record_atom_indices(record),
            )
        endpoint_directions.append(direction)

    handedness = float(
        np.dot(
            axis_unit,
            np.cross(
                endpoint_directions[0] / np.linalg.norm(endpoint_directions[0]),
                endpoint_directions[1] / np.linalg.norm(endpoint_directions[1]),
            ),
        )
    )
    sign = _normalized_geometry_sign(handedness, 1.0)
    if sign == 0:
        return None, _geometry_issue(
            "AX endpoint carrier triple product is degenerate within relative "
            f"tolerance {_GEOMETRY_RELATIVE_TOLERANCE:g}",
            record_index=record_index,
            atom_indices=_record_atom_indices(record),
        )
    return (
        _Level2Descriptor(
            _Level2DescriptorFamily.AXIAL,
            "even" if sign > 0 else "odd",
        ),
        None,
    )


def _gate4_coordinate_assignments(
    payload: StereoPayload,
    atom_count: int,
    coordinates: NDArray[np.float64],
    graph_report: StereoValidationReport,
) -> tuple[dict[int, _Level2Descriptor], tuple[StereoValidationIssue, ...]]:
    """Assign coordinates only for records accepted by the earlier gates."""

    if any(issue.record_index is None for issue in graph_report.issues):
        return {}, ()
    array, array_issue = _finite_coordinate_array(atom_count, coordinates)
    if array_issue is not None:
        return {}, (array_issue,)

    invalid_records = {
        issue.record_index
        for issue in graph_report.issues
        if issue.record_index is not None
    }
    assignments = {}
    issues = []
    for record_index, record in enumerate(payload.records):
        if record_index in invalid_records:
            continue
        parsed = _parse_level2_descriptor(record.descriptor).descriptor
        if parsed is None or parsed.family not in (
            _Level2DescriptorFamily.TETRAHEDRAL,
            _Level2DescriptorFamily.DOUBLE_BOND,
            _Level2DescriptorFamily.AXIAL,
            *_COORDINATION_FAMILIES,
        ):
            continue
        if parsed.family is _Level2DescriptorFamily.TETRAHEDRAL:
            assigned, geometry_issue = _tetrahedral_coordinate_descriptor(
                record,
                array,
                record_index=record_index,
            )
        elif parsed.family is _Level2DescriptorFamily.DOUBLE_BOND:
            assigned, geometry_issue = _double_bond_coordinate_descriptor(
                record,
                array,
                record_index=record_index,
            )
        elif parsed.family is _Level2DescriptorFamily.AXIAL:
            assigned, geometry_issue = _axial_coordinate_descriptor(
                record,
                array,
                record_index=record_index,
            )
        else:
            assigned, geometry_issue = _gate7_coordination_coordinate_descriptor(
                _gate7_site_from_record(record_index, record, parsed),
                array,
            )
        if geometry_issue is not None:
            issues.append(geometry_issue)
            continue
        assignments[record_index] = assigned

    issues.extend(
        _coordinate_group_consistency_issues(
            payload,
            assignments,
        )
    )
    return assignments, tuple(issues)
