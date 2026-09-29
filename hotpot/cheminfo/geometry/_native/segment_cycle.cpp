#include "segment_cycle.hpp"
#include "planar_predicates.hpp"
#include "vector_math.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <utility>


namespace hotpot::geometry {
namespace {

using detail::add_scaled;
using detail::all_finite;
using detail::dot;
using detail::finite;
using detail::norm;
using detail::projected_segment_polygon_contact;
using detail::subtract;


SegmentCycleFeature point_boundary_feature(
    const Point3& point,
    ArrayView<Point3> cycle,
    const PredicateTolerances& predicate_tolerances
) noexcept {
    for (const Point3& vertex : cycle) {
        if (norm(subtract(point, vertex)) <= predicate_tolerances.length) {
            return SegmentCycleFeature::CYCLE_VERTEX_CONTACT;
        }
    }
    return SegmentCycleFeature::CYCLE_EDGE_CONTACT;
}


SurfaceFamilyEvidence empty_evidence(bool enumeration_complete = false) noexcept {
    return {
        enumeration_complete,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
    };
}


SurfaceFamilyEvidence planar_evidence(PiercingState state) noexcept {
    return {
        true,
        1,
        1,
        0,
        0,
        state == PiercingState::PIERCES ? 1U : 0U,
        state == PiercingState::DOES_NOT_PIERCE ? 1U : 0U,
        state == PiercingState::UNDETERMINED ? 1U : 0U,
        0,
        0,
    };
}


void add_feature(
    std::vector<SegmentCycleFeature>& features,
    SegmentCycleFeature feature
) {
    if (std::find(features.begin(), features.end(), feature) == features.end()) {
        features.push_back(feature);
    }
}


void add_cause(
    std::vector<SegmentCycleIndeterminacy>& causes,
    SegmentCycleIndeterminacy cause
) {
    if (std::find(causes.begin(), causes.end(), cause) == causes.end()) {
        causes.push_back(cause);
    }
}


std::optional<SegmentCycleIndeterminacy> query_indeterminacy(
    const Segment3& segment,
    const PreparedPlanarCycle& cycle,
    PredicateTolerances& predicate_tolerances
) {
    const NumericTolerances& tolerances = cycle.tolerances;
    if (
        !finite(segment.start)
        || !finite(segment.end)
        || !all_finite(ArrayView<Point3>(cycle.coordinates))
    ) {
        return SegmentCycleIndeterminacy::NONFINITE_INPUT;
    }
    const double length_scale = segment_cycle_length_scale(
        segment,
        ArrayView<Point3>(cycle.coordinates)
    );
    if (!std::isfinite(length_scale)) {
        return SegmentCycleIndeterminacy::NUMERIC_BAND;
    }
    if (length_scale <= tolerances.absolute_length) {
        return SegmentCycleIndeterminacy::DEGENERATE_CYCLE;
    }
    predicate_tolerances = derive_predicate_tolerances(
        length_scale,
        tolerances
    );
    if (
        tolerances.predicate_guard_factor * predicate_tolerances.parameter
        >= 0.5
    ) {
        return SegmentCycleIndeterminacy::TOLERANCE_DOMAIN;
    }
    const double segment_length = norm(subtract(segment.end, segment.start));
    if (segment_length <= predicate_tolerances.length) {
        return SegmentCycleIndeterminacy::DEGENERATE_SEGMENT;
    }
    if (
        segment_length
        <= tolerances.predicate_guard_factor * predicate_tolerances.length
    ) {
        return SegmentCycleIndeterminacy::NUMERIC_BAND;
    }
    return std::nullopt;
}


SegmentCycleRelation undetermined_relation(
    const Segment3& segment,
    const PreparedPlanarCycle& cycle,
    SegmentCycleIndeterminacy cause
) {
    return {
        PiercingState::UNDETERMINED,
        {},
        {cause},
        std::nullopt,
        {},
        closest_cycle_edge(
            cycle,
            segment
        ),
        empty_evidence(),
    };
}


}  // namespace


std::optional<ClosestCycleEdge> closest_cycle_edge(
    const PreparedPlanarCycle& cycle,
    const Segment3& segment
) {
    const ArrayView<Point3> coordinates(cycle.coordinates);
    detail::require_cycle(coordinates);
    const NumericTolerances& tolerances = cycle.tolerances;
    if (
        !all_finite(coordinates)
        || !finite(segment.start)
        || !finite(segment.end)
    ) {
        return std::nullopt;
    }
    const double length_scale = segment_cycle_length_scale(
        segment,
        coordinates
    );
    if (
        !std::isfinite(length_scale)
        || length_scale <= tolerances.absolute_length
    ) {
        return std::nullopt;
    }
    const double length_tolerance = derive_predicate_tolerances(
        length_scale,
        tolerances
    ).length;
    std::vector<double> distances;
    distances.reserve(coordinates.size());
    double minimum = std::numeric_limits<double>::infinity();
    for (std::size_t index = 0; index < coordinates.size(); ++index) {
        const double distance = segment_segment_distance(
            {
                coordinates[index],
                coordinates[(index + 1) % coordinates.size()],
            },
            segment,
            tolerances
        );
        distances.push_back(distance);
        if (std::isfinite(distance)) {
            minimum = std::min(minimum, distance);
        }
    }
    if (!std::isfinite(minimum)) {
        return std::nullopt;
    }
    for (std::size_t index = 0; index < distances.size(); ++index) {
        if (
            std::isfinite(distances[index])
            && distances[index] <= minimum + length_tolerance
        ) {
            return ClosestCycleEdge{index, distances[index]};
        }
    }
    return std::nullopt;
}


SegmentCycleRelation determine_planar_segment_cycle_relation(
    const Segment3& segment,
    const PreparedPlanarCycle& cycle
) {
    detail::require_cycle(ArrayView<Point3>(cycle.coordinates));
    const NumericTolerances& tolerances = cycle.tolerances;
    PredicateTolerances predicate_tolerances{};
    const std::optional<SegmentCycleIndeterminacy> query_cause = (
        query_indeterminacy(segment, cycle, predicate_tolerances)
    );
    if (query_cause.has_value()) {
        return undetermined_relation(segment, cycle, *query_cause);
    }
    if (!cycle.has_planar_surface()) {
        const SegmentCycleIndeterminacy cause = (
            cycle.planarity.kind == PlanarityKind::DEGENERATE
            ? SegmentCycleIndeterminacy::DEGENERATE_CYCLE
            : SegmentCycleIndeterminacy::NUMERIC_BAND
        );
        return undetermined_relation(segment, cycle, cause);
    }

    const std::optional<ClosestCycleEdge> closest = closest_cycle_edge(
        cycle,
        segment
    );
    if (cycle.simplicity == PolygonSimplicity::SELF_INTERSECTING) {
        SurfaceFamilyEvidence evidence = empty_evidence(true);
        evidence.enumerated_surface_count = 1;
        evidence.proven_non_embedded_surface_count = 1;
        return {
            PiercingState::UNDETERMINED,
            {},
            {SegmentCycleIndeterminacy::SELF_INTERSECTION},
            CycleSurfaceModel::PLANAR_POLYGON,
            {},
            closest,
            evidence,
        };
    }
    if (cycle.simplicity == PolygonSimplicity::UNDETERMINED) {
        SurfaceFamilyEvidence evidence = empty_evidence(true);
        evidence.enumerated_surface_count = 1;
        evidence.construction_undetermined_count = 1;
        return {
            PiercingState::UNDETERMINED,
            {},
            {SegmentCycleIndeterminacy::NUMERIC_BAND},
            CycleSurfaceModel::PLANAR_POLYGON,
            {},
            closest,
            evidence,
        };
    }

    const Point3& normal = *cycle.planarity.normal;
    const Point3& origin = cycle.planarity.centroid;
    const Point3 direction = subtract(segment.end, segment.start);
    const double start_height = dot(normal, subtract(segment.start, origin));
    const double end_height = dot(normal, subtract(segment.end, origin));
    const double start_absolute = std::abs(start_height);
    const double end_absolute = std::abs(end_height);
    const double guard = tolerances.predicate_guard_factor;
    std::vector<SegmentCycleFeature> features;
    std::vector<SegmentCycleIndeterminacy> causes;
    std::vector<Point3> points;
    PiercingState state = PiercingState::DOES_NOT_PIERCE;

    if (
        start_absolute <= predicate_tolerances.length
        && end_absolute <= predicate_tolerances.length
    ) {
        const std::array<Point2, 2> projected_segment = {
            project_point_to_plane(segment.start, origin, normal),
            project_point_to_plane(segment.end, origin, normal),
        };
        const auto [has_contact, contact_undetermined] = (
            projected_segment_polygon_contact(
                projected_segment,
                ArrayView<Point2>(cycle.projection),
                predicate_tolerances,
                tolerances
            )
        );
        if (has_contact) {
            add_feature(features, SegmentCycleFeature::COPLANAR_CONTACT);
        } else if (contact_undetermined) {
            state = PiercingState::UNDETERMINED;
            add_cause(causes, SegmentCycleIndeterminacy::NUMERIC_BAND);
        }
    } else if (
        (
            predicate_tolerances.length < start_absolute
            && start_absolute <= guard * predicate_tolerances.length
        )
        || (
            predicate_tolerances.length < end_absolute
            && end_absolute <= guard * predicate_tolerances.length
        )
    ) {
        state = PiercingState::UNDETERMINED;
        add_cause(causes, SegmentCycleIndeterminacy::NUMERIC_BAND);
    } else {
        const double height_difference = start_height - end_height;
        const bool one_endpoint = (
            (
                start_absolute <= predicate_tolerances.length
                && end_absolute > guard * predicate_tolerances.length
            )
            || (
                end_absolute <= predicate_tolerances.length
                && start_absolute > guard * predicate_tolerances.length
            )
        );
        if (one_endpoint) {
            const Point3& point = (
                start_absolute <= predicate_tolerances.length
                ? segment.start
                : segment.end
            );
            const PointCycleLocation location = locate_projected_point(
                project_point_to_plane(point, origin, normal),
                ArrayView<Point2>(cycle.projection),
                predicate_tolerances,
                tolerances
            );
            if (
                location == PointCycleLocation::INTERIOR
                || location == PointCycleLocation::BOUNDARY
            ) {
                add_feature(features, SegmentCycleFeature::SEGMENT_ENDPOINT_CONTACT);
                points.push_back(point);
            }
            if (location == PointCycleLocation::BOUNDARY) {
                add_feature(
                    features,
                    point_boundary_feature(
                        point,
                        ArrayView<Point3>(cycle.coordinates),
                        predicate_tolerances
                    )
                );
            } else if (location == PointCycleLocation::UNDETERMINED) {
                state = PiercingState::UNDETERMINED;
                add_cause(causes, SegmentCycleIndeterminacy::NUMERIC_BAND);
            }
        } else if (std::abs(height_difference) <= predicate_tolerances.length) {
            // The segment and plane are stably separated.
        } else if (
            std::abs(height_difference)
            <= guard * predicate_tolerances.length
        ) {
            state = PiercingState::UNDETERMINED;
            add_cause(causes, SegmentCycleIndeterminacy::NUMERIC_BAND);
        } else {
            const double parameter = start_height / height_difference;
            const Point3 point = add_scaled(segment.start, direction, parameter);
            const PointCycleLocation location = locate_projected_point(
                project_point_to_plane(point, origin, normal),
                ArrayView<Point2>(cycle.projection),
                predicate_tolerances,
                tolerances
            );
            const double parameter_tolerance = predicate_tolerances.parameter;
            const bool parameter_inside = (
                guard * parameter_tolerance < parameter
                && parameter < 1.0 - guard * parameter_tolerance
            );
            const bool parameter_endpoint = (
                std::abs(parameter) <= parameter_tolerance
                || std::abs(1.0 - parameter) <= parameter_tolerance
            );
            const bool parameter_outside = (
                parameter < -guard * parameter_tolerance
                || parameter > 1.0 + guard * parameter_tolerance
            );
            if (location == PointCycleLocation::INTERIOR) {
                if (parameter_inside) {
                    state = PiercingState::PIERCES;
                    add_feature(features, SegmentCycleFeature::TRANSVERSE_INTERIOR);
                    points.push_back(point);
                } else if (parameter_endpoint) {
                    add_feature(
                        features,
                        SegmentCycleFeature::SEGMENT_ENDPOINT_CONTACT
                    );
                    points.push_back(point);
                } else if (parameter_outside) {
                    add_feature(
                        features,
                        SegmentCycleFeature::LINE_EXTENSION_INTERIOR
                    );
                } else {
                    state = PiercingState::UNDETERMINED;
                    add_cause(causes, SegmentCycleIndeterminacy::NUMERIC_BAND);
                }
            } else if (location == PointCycleLocation::BOUNDARY) {
                if (parameter_inside || parameter_endpoint) {
                    add_feature(
                        features,
                        point_boundary_feature(
                            point,
                            ArrayView<Point3>(cycle.coordinates),
                            predicate_tolerances
                        )
                    );
                    if (parameter_endpoint) {
                        add_feature(
                            features,
                            SegmentCycleFeature::SEGMENT_ENDPOINT_CONTACT
                        );
                    }
                    points.push_back(point);
                } else if (!parameter_outside) {
                    state = PiercingState::UNDETERMINED;
                    add_cause(causes, SegmentCycleIndeterminacy::NUMERIC_BAND);
                }
            } else if (location == PointCycleLocation::UNDETERMINED) {
                state = PiercingState::UNDETERMINED;
                add_cause(causes, SegmentCycleIndeterminacy::NUMERIC_BAND);
            }
        }
    }

    return {
        state,
        std::move(features),
        std::move(causes),
        CycleSurfaceModel::PLANAR_POLYGON,
        std::move(points),
        closest,
        planar_evidence(state),
    };
}


std::vector<SegmentCycleRelation> planar_segment_cycle_relations(
    ArrayView<Segment3> segments,
    const PreparedPlanarCycle& cycle
) {
    detail::require_cycle(ArrayView<Point3>(cycle.coordinates));
    std::vector<SegmentCycleRelation> relations;
    relations.reserve(segments.size());
    for (const Segment3& segment : segments) {
        relations.push_back(determine_planar_segment_cycle_relation(
            segment,
            cycle
        ));
    }
    return relations;
}


std::vector<SegmentCycleScreening> planar_segment_cycle_screenings(
    ArrayView<Segment3> segments,
    const PreparedPlanarCycle& cycle
) {
    detail::require_cycle(ArrayView<Point3>(cycle.coordinates));
    std::vector<SegmentCycleScreening> screenings;
    screenings.reserve(segments.size());
    for (const Segment3& segment : segments) {
        PredicateTolerances predicate_tolerances{};
        const std::optional<SegmentCycleIndeterminacy> cause = query_indeterminacy(
            segment,
            cycle,
            predicate_tolerances
        );
        if (
            !cause.has_value()
            && cycle.has_simple_planar_surface()
        ) {
            const std::array<Point3, 2> endpoints = {
                segment.start,
                segment.end,
            };
            if (aabb_stably_separated(
                    aabb_bounds(ArrayView<Point3>(
                        endpoints.data(),
                        endpoints.size()
                    )),
                    cycle.bounds,
                    predicate_tolerances.aabb
                )) {
                screenings.push_back({
                    PiercingState::DOES_NOT_PIERCE,
                    std::nullopt,
                    true,
                    true,
                });
                continue;
            }
        }
        SegmentCycleRelation relation = determine_planar_segment_cycle_relation(
            segment,
            cycle
        );
        const bool surface_complete = relation.surface_evidence.enumeration_complete;
        const PiercingState state = relation.state;
        screenings.push_back({
            state,
            std::move(relation),
            false,
            surface_complete,
        });
    }
    return screenings;
}


}  // namespace hotpot::geometry
