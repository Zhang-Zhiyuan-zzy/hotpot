#include "nonplanar_segment.hpp"

#include "planar_predicates.hpp"
#include "triangle_predicates.hpp"
#include "vector_math.hpp"

#include <algorithm>
#include <array>
#include <optional>
#include <utility>
#include <vector>


namespace hotpot::geometry {
namespace {


enum class SurfaceSegmentState : std::uint8_t {
    INTERSECTING,
    NON_PIERCING,
    EVALUATION_UNDETERMINED,
};


struct SurfaceSegmentFacts {
    SurfaceSegmentState state;
    bool segment_triangle_budget_exhausted;
};


struct NonplanarRelationDetails {
    std::vector<SegmentCycleFeature> features;
    std::vector<SegmentCycleIndeterminacy> causes;
    std::vector<Point3> points;
};


struct NonplanarRelationFacts {
    PiercingState state;
    std::optional<CycleSurfaceModel> surface_model;
    SurfaceFamilyEvidence surface_evidence;
};


struct SegmentTriangleCounters {
    std::size_t tests = 0;
};


using detail::dot;
using detail::norm;
using detail::projected_segment_polygon_contact;
using detail::segment_triangle_relation;
using detail::subtract;
using detail::TriangleHit;
using detail::TriangleHitKind;


template <typename Value>
void add_unique(std::vector<Value>& values, Value value) {
    if (std::find(values.begin(), values.end(), value) == values.end()) {
        values.push_back(value);
    }
}


void record_feature(
    NonplanarRelationDetails* details,
    SegmentCycleFeature feature
) {
    if (details != nullptr) {
        add_unique(details->features, feature);
    }
}


void record_cause(
    NonplanarRelationDetails* details,
    SegmentCycleIndeterminacy cause
) {
    if (details != nullptr) {
        add_unique(details->causes, cause);
    }
}


void record_point(
    NonplanarRelationDetails* details,
    const Point3& point
) {
    if (details != nullptr) {
        details->points.push_back(point);
    }
}


double point_segment_distance_exact(
    const Point3& point,
    const Point3& start,
    const Point3& end
) noexcept {
    const Point3 direction = subtract(end, start);
    const double squared_length = dot(direction, direction);
    if (squared_length == 0.0) {
        return norm(subtract(point, start));
    }
    const double parameter = std::clamp(
        dot(subtract(point, start), direction) / squared_length,
        0.0,
        1.0
    );
    const Point3 closest = {
        start[0] + parameter * direction[0],
        start[1] + parameter * direction[1],
        start[2] + parameter * direction[2],
    };
    return norm(subtract(point, closest));
}


bool point_near_internal_edge(
    const Point3& point,
    const PreparedSurfaceGeometry& surface,
    ArrayView<Point3> cycle,
    double tolerance
) noexcept {
    for (const EdgeIndices& edge : surface.internal_edges) {
        if (
            point_segment_distance_exact(
                point,
                cycle[edge[0]],
                cycle[edge[1]]
            ) <= tolerance
        ) {
            return true;
        }
    }
    return false;
}


bool point_on_cycle_boundary(
    const Point3& point,
    ArrayView<Point3> cycle,
    double tolerance
) noexcept {
    for (std::size_t index = 0; index < cycle.size(); ++index) {
        if (
            point_segment_distance_exact(
                point,
                cycle[index],
                cycle[(index + 1) % cycle.size()]
            ) <= tolerance
        ) {
            return true;
        }
    }
    return false;
}


SegmentCycleFeature point_boundary_feature(
    const Point3& point,
    ArrayView<Point3> cycle,
    double tolerance
) noexcept {
    for (const Point3& vertex : cycle) {
        if (norm(subtract(point, vertex)) <= tolerance) {
            return SegmentCycleFeature::CYCLE_VERTEX_CONTACT;
        }
    }
    return SegmentCycleFeature::CYCLE_EDGE_CONTACT;
}


std::vector<Point3> merge_points(
    const std::vector<Point3>& points,
    double tolerance
) {
    std::vector<Point3> merged;
    merged.reserve(points.size());
    for (const Point3& point : points) {
        const bool already_present = std::any_of(
            merged.begin(),
            merged.end(),
            [&point, tolerance](const Point3& current) {
                return norm(subtract(point, current)) <= tolerance;
            }
        );
        if (!already_present) {
            merged.push_back(point);
        }
    }
    return merged;
}


std::pair<bool, bool> coplanar_segment_triangle_contact(
    const Segment3& segment,
    const PreparedTriangleGeometry& triangle,
    const PredicateTolerances& predicate_tolerances,
    const NumericTolerances& tolerances
) {
    Point3 unit_normal = triangle.normal;
    for (double& value : unit_normal) {
        value /= triangle.normal_length;
    }
    const std::array<Point2, 3> projected_triangle = {{
        project_point_to_plane(
            triangle.coordinates[0], triangle.coordinates[0], unit_normal
        ),
        project_point_to_plane(
            triangle.coordinates[1], triangle.coordinates[0], unit_normal
        ),
        project_point_to_plane(
            triangle.coordinates[2], triangle.coordinates[0], unit_normal
        ),
    }};
    const std::array<Point2, 2> projected_segment = {{
        project_point_to_plane(
            segment.start, triangle.coordinates[0], unit_normal
        ),
        project_point_to_plane(
            segment.end, triangle.coordinates[0], unit_normal
        ),
    }};
    return projected_segment_polygon_contact(
        projected_segment,
        ArrayView<Point2>(
            projected_triangle.data(), projected_triangle.size()
        ),
        predicate_tolerances,
        tolerances
    );
}


SurfaceSegmentFacts determine_surface_segment_relation(
    const Segment3& segment,
    const PreparedSurfaceGeometry& surface,
    const PreparedNonplanarSurfaceFamily& family,
    const PredicateTolerances& predicate_tolerances,
    SegmentTriangleCounters& counters,
    NonplanarRelationDetails* details
) {
    bool confirmed_intersection = false;
    bool evaluation_undetermined = false;
    bool segment_triangle_budget_exhausted = false;
    const double guard = family.tolerances().predicate_guard_factor;
    const ArrayView<Point3> cycle(family.coordinates());

    for (const std::size_t triangle_position : surface.triangle_positions) {
        if (
            counters.tests
            >= family.limits().maximum_segment_triangle_tests
        ) {
            record_cause(
                details,
                SegmentCycleIndeterminacy::INCOMPLETE_SURFACE_FAMILY
            );
            evaluation_undetermined = true;
            segment_triangle_budget_exhausted = true;
            break;
        }
        ++counters.tests;
        const PreparedTriangleGeometry& triangle =
            family.unique_triangles()[triangle_position];
        const TriangleHit hit = segment_triangle_relation(
            segment,
            triangle,
            predicate_tolerances,
            family.tolerances()
        );
        if (
            hit.kind == TriangleHitKind::STRICT_INTERIOR
            && hit.point.has_value()
        ) {
            if (point_near_internal_edge(
                    *hit.point,
                    surface,
                    cycle,
                    guard * predicate_tolerances.length
                )) {
                evaluation_undetermined = true;
                record_cause(
                    details,
                    SegmentCycleIndeterminacy::NUMERIC_BAND
                );
            } else {
                confirmed_intersection = true;
                record_feature(
                    details,
                    SegmentCycleFeature::TRANSVERSE_INTERIOR
                );
                record_point(details, *hit.point);
            }
        } else if (hit.kind == TriangleHitKind::LINE_EXTENSION_INTERIOR) {
            record_feature(
                details,
                SegmentCycleFeature::LINE_EXTENSION_INTERIOR
            );
        } else if (
            hit.kind == TriangleHitKind::TRIANGLE_BOUNDARY
            && hit.point.has_value()
        ) {
            if (point_near_internal_edge(
                    *hit.point,
                    surface,
                    cycle,
                    guard * predicate_tolerances.length
                )) {
                evaluation_undetermined = true;
                record_cause(
                    details,
                    SegmentCycleIndeterminacy::NUMERIC_BAND
                );
            } else {
                if (details != nullptr) {
                    record_feature(
                        details,
                        point_boundary_feature(
                            *hit.point,
                            cycle,
                            predicate_tolerances.length
                        )
                    );
                    record_point(details, *hit.point);
                }
            }
        } else if (hit.kind == TriangleHitKind::SEGMENT_ENDPOINT) {
            if (hit.point.has_value()) {
                if (point_on_cycle_boundary(
                        *hit.point,
                        cycle,
                        predicate_tolerances.length
                    )) {
                    record_feature(
                        details,
                        SegmentCycleFeature::SEGMENT_ENDPOINT_CONTACT
                    );
                    if (details != nullptr) {
                        record_feature(
                            details,
                            point_boundary_feature(
                                *hit.point,
                                cycle,
                                predicate_tolerances.length
                            )
                        );
                        record_point(details, *hit.point);
                    }
                } else if (point_near_internal_edge(
                        *hit.point,
                        surface,
                        cycle,
                        guard * predicate_tolerances.length
                    )) {
                    evaluation_undetermined = true;
                    record_cause(
                        details,
                        SegmentCycleIndeterminacy::NUMERIC_BAND
                    );
                } else {
                    record_feature(
                        details,
                        SegmentCycleFeature::SEGMENT_ENDPOINT_CONTACT
                    );
                    record_point(details, *hit.point);
                }
            }
        } else if (hit.kind == TriangleHitKind::COPLANAR) {
            const auto [has_contact, contact_undetermined] =
                coplanar_segment_triangle_contact(
                    segment,
                    triangle,
                    predicate_tolerances,
                    family.tolerances()
                );
            if (has_contact) {
                record_feature(
                    details,
                    SegmentCycleFeature::COPLANAR_CONTACT
                );
            } else if (contact_undetermined) {
                evaluation_undetermined = true;
                record_cause(
                    details,
                    SegmentCycleIndeterminacy::NUMERIC_BAND
                );
            }
        } else if (hit.kind == TriangleHitKind::DEGENERATE) {
            evaluation_undetermined = true;
            record_cause(
                details,
                SegmentCycleIndeterminacy::DEGENERATE_TRIANGLE
            );
        } else if (hit.kind == TriangleHitKind::UNDETERMINED) {
            evaluation_undetermined = true;
            record_cause(details, SegmentCycleIndeterminacy::NUMERIC_BAND);
        }
    }

    SurfaceSegmentState state = SurfaceSegmentState::NON_PIERCING;
    if (confirmed_intersection) {
        state = SurfaceSegmentState::INTERSECTING;
    } else if (evaluation_undetermined) {
        state = SurfaceSegmentState::EVALUATION_UNDETERMINED;
    }
    return {
        state,
        segment_triangle_budget_exhausted,
    };
}


SegmentCycleIndeterminacy map_surface_cause(NonplanarSurfaceCause cause) {
    if (cause == NonplanarSurfaceCause::INCOMPLETE_SURFACE_FAMILY) {
        return SegmentCycleIndeterminacy::INCOMPLETE_SURFACE_FAMILY;
    }
    return SegmentCycleIndeterminacy::SURFACE_CONSTRUCTION;
}


SurfaceFamilyEvidence empty_evidence() noexcept {
    return {false, 0, 0, 0, 0, 0, 0, 0, 0, 0};
}


NonplanarRelationFacts evaluate_prepared_relation(
    const Segment3& segment,
    const PreparedNonplanarSurfaceFamily& family,
    const detail::SegmentCycleQuery& query,
    NonplanarRelationDetails* details
) {
    if (query.cause.has_value()) {
        record_cause(details, *query.cause);
        return {
            PiercingState::UNDETERMINED,
            std::nullopt,
            empty_evidence(),
        };
    }
    const PredicateTolerances& predicate_tolerances =
        *query.predicate_tolerances;
    bool enumeration_complete = family.enumeration_complete();
    const std::size_t embedded = family.embedded_surface_indices().size();
    std::size_t intersecting = 0;
    std::size_t non_piercing = 0;
    std::size_t evaluation_undetermined = 0;
    for (const NonplanarSurfaceCause cause : family.causes()) {
        record_cause(details, map_surface_cause(cause));
    }
    SegmentTriangleCounters counters;

    for (std::size_t ordinal = 0; ordinal < embedded; ++ordinal) {
        const std::size_t surface_index =
            family.embedded_surface_indices()[ordinal];
        const SurfaceSegmentFacts result = determine_surface_segment_relation(
            segment,
            family.surfaces()[surface_index],
            family,
            predicate_tolerances,
            counters,
            details
        );
        if (result.state == SurfaceSegmentState::INTERSECTING) {
            ++intersecting;
        } else if (result.state == SurfaceSegmentState::NON_PIERCING) {
            ++non_piercing;
        } else {
            ++evaluation_undetermined;
        }
        if (result.segment_triangle_budget_exhausted) {
            enumeration_complete = false;
            record_cause(
                details,
                SegmentCycleIndeterminacy::INCOMPLETE_SURFACE_FAMILY
            );
            evaluation_undetermined += embedded - ordinal - 1;
            break;
        }
    }

    if (intersecting > 0 && non_piercing > 0) {
        record_cause(
            details,
            SegmentCycleIndeterminacy::SURFACE_DISAGREEMENT
        );
    }
    const SurfaceFamilyEvidence evidence{
        enumeration_complete,
        family.enumerated_surface_count(),
        embedded,
        family.proven_non_embedded_surface_count(),
        family.construction_undetermined_count(),
        intersecting,
        non_piercing,
        evaluation_undetermined,
        counters.tests,
        family.triangle_pair_tests_used(),
    };

    PiercingState state = PiercingState::UNDETERMINED;
    if (
        enumeration_complete
        && family.construction_undetermined_count() == 0
        && evaluation_undetermined == 0
        && embedded > 0
        && intersecting == embedded
    ) {
        state = PiercingState::PIERCES;
    } else if (
        enumeration_complete
        && family.construction_undetermined_count() == 0
        && evaluation_undetermined == 0
        && embedded > 0
        && non_piercing == embedded
    ) {
        state = PiercingState::DOES_NOT_PIERCE;
    } else if (details != nullptr && details->causes.empty()) {
        record_cause(
            details,
            SegmentCycleIndeterminacy::SURFACE_DISAGREEMENT
        );
    }

    return {
        state,
        CycleSurfaceModel::VERTEX_TRIANGULATION_FAMILY,
        evidence,
    };
}


SegmentCycleRelation materialize_prepared_relation(
    const Segment3& segment,
    const PreparedNonplanarSurfaceFamily& family,
    const detail::SegmentCycleQuery& query
) {
    NonplanarRelationDetails details;
    const NonplanarRelationFacts facts = evaluate_prepared_relation(
        segment,
        family,
        query,
        &details
    );
    return {
        facts.state,
        std::move(details.features),
        std::move(details.causes),
        facts.surface_model,
        merge_points(
            details.points,
            query.predicate_tolerances.has_value()
                ? query.predicate_tolerances->merge
                : 0.0
        ),
        detail::closest_cycle_edge(
            ArrayView<Point3>(family.coordinates()),
            segment,
            family.tolerances()
        ),
        facts.surface_evidence,
    };
}


}  // namespace


SegmentCycleRelation determine_nonplanar_segment_cycle_relation(
    const Segment3& segment,
    const PreparedNonplanarSurfaceFamily& family
) {
    const ArrayView<Point3> cycle(family.coordinates());
    detail::require_cycle(cycle);
    const detail::SegmentCycleQuery query =
        detail::prepare_segment_cycle_query(
            segment,
            cycle,
            family.tolerances()
        );
    return materialize_prepared_relation(segment, family, query);
}


SegmentCycleScreening screen_nonplanar_segment_cycle(
    const Segment3& segment,
    const PreparedNonplanarSurfaceFamily& family,
    const Aabb& cycle_bounds,
    bool materialize_relation
) {
    const ArrayView<Point3> cycle(family.coordinates());
    detail::require_cycle(cycle);
    const detail::SegmentCycleQuery query =
        detail::prepare_segment_cycle_query(
            segment,
            cycle,
            family.tolerances()
        );
    const bool surface_is_valid = (
        family.enumeration_complete()
        && family.construction_undetermined_count() == 0
        && !family.embedded_surface_indices().empty()
    );
    if (!query.cause.has_value() && surface_is_valid) {
        const std::array<Point3, 2> endpoints = {
            segment.start,
            segment.end,
        };
        if (aabb_stably_separated(
                aabb_bounds(ArrayView<Point3>(
                    endpoints.data(),
                    endpoints.size()
                )),
                cycle_bounds,
                query.predicate_tolerances->aabb
            )) {
            return {
                PiercingState::DOES_NOT_PIERCE,
                std::nullopt,
                true,
                true,
            };
        }
    }
    if (materialize_relation) {
        SegmentCycleRelation relation = materialize_prepared_relation(
            segment,
            family,
            query
        );
        const PiercingState state = relation.state;
        const bool surface_complete =
            relation.surface_evidence.enumeration_complete;
        return {
            state,
            std::move(relation),
            false,
            surface_complete,
        };
    }
    const NonplanarRelationFacts facts = evaluate_prepared_relation(
        segment,
        family,
        query,
        nullptr
    );
    return {
        facts.state,
        std::nullopt,
        false,
        facts.surface_evidence.enumeration_complete,
    };
}


SegmentCycleScreening screen_nonplanar_segment_cycle(
    const Segment3& segment,
    const PreparedNonplanarSurfaceFamily& family,
    bool materialize_relation
) {
    return screen_nonplanar_segment_cycle(
        segment,
        family,
        aabb_bounds(ArrayView<Point3>(family.coordinates())),
        materialize_relation
    );
}


std::vector<SegmentCycleRelation> nonplanar_segment_cycle_relations(
    ArrayView<Segment3> segments,
    const PreparedNonplanarSurfaceFamily& family
) {
    const ArrayView<Point3> cycle(family.coordinates());
    detail::require_cycle(cycle);
    std::vector<SegmentCycleRelation> relations;
    relations.reserve(segments.size());
    for (const Segment3& segment : segments) {
        const detail::SegmentCycleQuery query =
            detail::prepare_segment_cycle_query(
                segment,
                cycle,
                family.tolerances()
            );
        relations.push_back(materialize_prepared_relation(
            segment,
            family,
            query
        ));
    }
    return relations;
}


std::vector<SegmentCycleScreening> nonplanar_segment_cycle_screenings(
    ArrayView<Segment3> segments,
    const PreparedNonplanarSurfaceFamily& family
) {
    const ArrayView<Point3> cycle(family.coordinates());
    detail::require_cycle(cycle);
    const Aabb cycle_bounds = aabb_bounds(cycle);
    std::vector<SegmentCycleScreening> screenings;
    screenings.reserve(segments.size());
    for (const Segment3& segment : segments) {
        screenings.push_back(screen_nonplanar_segment_cycle(
            segment,
            family,
            cycle_bounds,
            true
        ));
    }
    return screenings;
}


}  // namespace hotpot::geometry
