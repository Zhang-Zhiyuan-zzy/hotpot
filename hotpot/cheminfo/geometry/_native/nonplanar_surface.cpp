#include "nonplanar_surface.hpp"

#include "planar_predicates.hpp"
#include "triangle_predicates.hpp"
#include "vector_math.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <map>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <utility>


namespace hotpot::geometry {
namespace {


using SurfaceIndices = std::vector<TriangleIndices>;
using Triangulations = std::vector<SurfaceIndices>;


struct CycleTopology {
    std::size_t surface_count;
    std::vector<TriangleIndices> unique_triangles;
    std::vector<PreparedSurfaceGeometry> surfaces;
};


struct SurfaceCounters {
    std::size_t triangle_pair_tests = 0;
    bool triangle_pair_budget_exhausted = false;
};


struct SaturatingCount {
    std::size_t value;
    bool overflowed;
};


using detail::all_finite;
using detail::cross;
using detail::dot;
using detail::finite;
using detail::norm;
using detail::point_segment_distance_2d;
using detail::segment_triangle_relation;
using detail::subtract;
using detail::TriangleHit;
using detail::TriangleHitKind;


bool contains_index(
    const std::vector<std::size_t>& indices,
    std::size_t target
) noexcept {
    return std::find(indices.begin(), indices.end(), target) != indices.end();
}


bool contains_index(
    const TriangleIndices& indices,
    std::size_t target
) noexcept {
    return std::find(indices.begin(), indices.end(), target) != indices.end();
}


void add_cause(
    std::vector<NonplanarSurfaceCause>& causes,
    NonplanarSurfaceCause cause
) {
    if (std::find(causes.begin(), causes.end(), cause) == causes.end()) {
        causes.push_back(cause);
    }
}


std::size_t saturating_product(
    std::size_t first,
    std::size_t second,
    bool& overflowed
) noexcept {
    if (
        first != 0
        && second > std::numeric_limits<std::size_t>::max() / first
    ) {
        overflowed = true;
        return std::numeric_limits<std::size_t>::max();
    }
    return first * second;
}


std::vector<SaturatingCount> catalan_counts(std::size_t maximum_order) {
    std::vector<SaturatingCount> counts(maximum_order + 1, {0, false});
    counts[0] = {1, false};
    for (std::size_t current = 1; current <= maximum_order; ++current) {
        std::size_t count = 0;
        bool overflowed = false;
        for (std::size_t left = 0; left < current; ++left) {
            const std::size_t right = current - 1 - left;
            overflowed = (
                overflowed
                || counts[left].overflowed
                || counts[right].overflowed
            );
            const std::size_t product = saturating_product(
                counts[left].value,
                counts[right].value,
                overflowed
            );
            if (product > std::numeric_limits<std::size_t>::max() - count) {
                overflowed = true;
                count = std::numeric_limits<std::size_t>::max();
                break;
            }
            count += product;
        }
        counts[current] = {count, overflowed};
    }
    return counts;
}


std::size_t interval_catalan_order(
    std::size_t first,
    std::size_t last
) noexcept {
    return last - first < 2 ? 0 : last - first - 1;
}


SurfaceIndices triangulation_at(
    std::size_t first,
    std::size_t last,
    std::size_t rank,
    const std::vector<SaturatingCount>& counts
) {
    if (last - first < 2) {
        return {};
    }
    for (std::size_t middle = first + 1; middle < last; ++middle) {
        const std::size_t left_count = counts[
            interval_catalan_order(first, middle)
        ].value;
        const std::size_t right_count = counts[
            interval_catalan_order(middle, last)
        ].value;
        bool block_overflowed = false;
        const std::size_t block_count = saturating_product(
            left_count, right_count, block_overflowed
        );
        if (!block_overflowed && rank >= block_count) {
            rank -= block_count;
            continue;
        }

        SurfaceIndices left = triangulation_at(
            first, middle, rank / right_count, counts
        );
        SurfaceIndices right = triangulation_at(
            middle, last, rank % right_count, counts
        );
        left.reserve(left.size() + right.size() + 1);
        left.insert(left.end(), right.begin(), right.end());
        left.push_back({first, middle, last});
        return left;
    }
    throw std::logic_error("triangulation rank exceeds Catalan count");
}


Triangulations enumerate_cycle_triangulations(
    std::size_t vertex_count,
    std::size_t maximum_surface_count,
    const std::vector<SaturatingCount>& counts
) {
    const SaturatingCount expected = counts[vertex_count - 2];
    const std::size_t count = expected.overflowed
        ? maximum_surface_count
        : std::min(expected.value, maximum_surface_count);
    Triangulations triangulations;
    triangulations.reserve(count);
    for (std::size_t rank = 0; rank < count; ++rank) {
        triangulations.push_back(
            triangulation_at(0, vertex_count - 1, rank, counts)
        );
    }
    return triangulations;
}


EdgeIndices ordered_edge(std::size_t first, std::size_t second) noexcept {
    return first < second
        ? EdgeIndices{first, second}
        : EdgeIndices{second, first};
}


bool is_cycle_edge(const EdgeIndices& edge, std::size_t vertex_count) noexcept {
    for (std::size_t index = 0; index < vertex_count; ++index) {
        if (edge == ordered_edge(index, (index + 1) % vertex_count)) {
            return true;
        }
    }
    return false;
}


std::vector<EdgeIndices> surface_internal_edges(
    const SurfaceIndices& surface,
    std::size_t vertex_count
) {
    std::vector<std::pair<EdgeIndices, std::size_t>> counts;
    for (const TriangleIndices& triangle : surface) {
        for (std::size_t index = 0; index < 3; ++index) {
            const EdgeIndices edge = ordered_edge(
                triangle[index], triangle[(index + 1) % 3]
            );
            const auto existing = std::find_if(
                counts.begin(),
                counts.end(),
                [&edge](const auto& item) { return item.first == edge; }
            );
            if (existing == counts.end()) {
                counts.push_back({edge, 1});
            } else {
                ++existing->second;
            }
        }
    }

    std::vector<EdgeIndices> internal_edges;
    for (const auto& [edge, count] : counts) {
        if (count == 2 && !is_cycle_edge(edge, vertex_count)) {
            internal_edges.push_back(edge);
        }
    }
    return internal_edges;
}


std::vector<std::size_t> shared_simplex(
    const TriangleIndices& first,
    const TriangleIndices& second
) {
    std::vector<std::size_t> shared;
    for (const std::size_t index : first) {
        if (contains_index(second, index)) {
            shared.push_back(index);
        }
    }
    std::sort(shared.begin(), shared.end());
    return shared;
}


CycleTopology prepare_cycle_topology(
    std::size_t vertex_count,
    std::size_t maximum_surface_count,
    const std::vector<SaturatingCount>& counts
) {
    CycleTopology topology;
    const Triangulations triangulations = enumerate_cycle_triangulations(
        vertex_count, maximum_surface_count, counts
    );
    topology.surface_count = triangulations.size();
    for (const SurfaceIndices& surface : triangulations) {
        for (const TriangleIndices& triangle : surface) {
            if (
                std::find(
                    topology.unique_triangles.begin(),
                    topology.unique_triangles.end(),
                    triangle
                ) == topology.unique_triangles.end()
            ) {
                topology.unique_triangles.push_back(triangle);
            }
        }
    }

    topology.surfaces.reserve(triangulations.size());
    for (const SurfaceIndices& surface : triangulations) {
        PreparedSurfaceGeometry prepared;
        prepared.internal_edges = surface_internal_edges(surface, vertex_count);
        prepared.triangle_positions.reserve(surface.size());
        for (const TriangleIndices& triangle : surface) {
            const auto position = std::find(
                topology.unique_triangles.begin(),
                topology.unique_triangles.end(),
                triangle
            );
            prepared.triangle_positions.push_back(
                static_cast<std::size_t>(
                    std::distance(topology.unique_triangles.begin(), position)
                )
            );
        }
        for (std::size_t first = 0; first < surface.size(); ++first) {
            for (std::size_t second = first + 1; second < surface.size(); ++second) {
                prepared.triangle_pairs.push_back({first, second});
                prepared.shared_simplices.push_back(
                    shared_simplex(surface[first], surface[second])
                );
            }
        }
        topology.surfaces.push_back(std::move(prepared));
    }
    return topology;
}


std::shared_ptr<const CycleTopology> cached_cycle_topology(
    std::size_t vertex_count,
    std::size_t maximum_surface_count,
    const std::vector<SaturatingCount>& counts
) {
    using CacheKey = std::pair<std::size_t, std::size_t>;
    static std::mutex cache_mutex;
    static std::map<CacheKey, std::shared_ptr<const CycleTopology>> cache;

    const CacheKey key = {vertex_count, maximum_surface_count};
    std::lock_guard<std::mutex> lock(cache_mutex);
    const auto cached = cache.find(key);
    if (cached != cache.end()) {
        return cached->second;
    }
    auto topology = std::make_shared<const CycleTopology>(
        prepare_cycle_topology(vertex_count, maximum_surface_count, counts)
    );
    cache.emplace(key, topology);
    return topology;
}


PreparedTriangleGeometry prepare_triangle(
    ArrayView<Point3> cycle,
    const TriangleIndices& indices
) {
    const std::array<Point3, 3> coordinates = {
        cycle[indices[0]],
        cycle[indices[1]],
        cycle[indices[2]],
    };
    const Point3 normal = cross(
        subtract(coordinates[1], coordinates[0]),
        subtract(coordinates[2], coordinates[0])
    );
    const std::array<Segment3, 3> edges = {{
        {coordinates[0], coordinates[1]},
        {coordinates[1], coordinates[2]},
        {coordinates[2], coordinates[0]},
    }};
    return {
        indices,
        coordinates,
        normal,
        norm(normal),
        aabb_bounds(ArrayView<Point3>(coordinates.data(), coordinates.size())),
        edges,
    };
}


std::array<Point2, 3> project_triangle(
    const PreparedTriangleGeometry& triangle,
    const Point3& origin,
    const Point3& unit_normal
) noexcept {
    return {{
        project_point_to_plane(triangle.coordinates[0], origin, unit_normal),
        project_point_to_plane(triangle.coordinates[1], origin, unit_normal),
        project_point_to_plane(triangle.coordinates[2], origin, unit_normal),
    }};
}


SurfaceEmbeddingState coplanar_triangle_pair_state(
    const PreparedTriangleGeometry& first,
    const PreparedTriangleGeometry& second,
    const PredicateTolerances& predicate_tolerances,
    const NumericTolerances& tolerances,
    const std::vector<std::size_t>& shared
) {
    Point3 unit_normal = first.normal;
    for (double& value : unit_normal) {
        value /= first.normal_length;
    }
    const std::array<Point2, 3> first_projected = project_triangle(
        first, first.coordinates[0], unit_normal
    );
    const std::array<Point2, 3> second_projected = project_triangle(
        second, first.coordinates[0], unit_normal
    );
    const double guard = tolerances.predicate_guard_factor;

    for (std::size_t first_edge_index = 0; first_edge_index < 3; ++first_edge_index) {
        const std::size_t first_a = first.indices[first_edge_index];
        const std::size_t first_b = first.indices[(first_edge_index + 1) % 3];
        for (
            std::size_t second_edge_index = 0;
            second_edge_index < 3;
            ++second_edge_index
        ) {
            const std::size_t second_a = second.indices[second_edge_index];
            const std::size_t second_b = second.indices[
                (second_edge_index + 1) % 3
            ];
            std::size_t shared_edge_vertex_count = 0;
            shared_edge_vertex_count += (
                first_a == second_a || first_a == second_b
            );
            shared_edge_vertex_count += (
                first_b == second_a || first_b == second_b
            );
            if (shared_edge_vertex_count == 2) {
                continue;
            }
            const Point2& first_start = first_projected[first_edge_index];
            const Point2& first_end = first_projected[
                (first_edge_index + 1) % 3
            ];
            const Point2& second_start = second_projected[second_edge_index];
            const Point2& second_end = second_projected[
                (second_edge_index + 1) % 3
            ];
            if (detail::aabb_stably_separated_2d(
                    first_start,
                    first_end,
                    second_start,
                    second_end,
                    predicate_tolerances.aabb
                )) {
                continue;
            }
            const std::array<double, 4> orientations = {
                detail::orient2d(first_start, first_end, second_start),
                detail::orient2d(first_start, first_end, second_end),
                detail::orient2d(second_start, second_end, first_start),
                detail::orient2d(second_start, second_end, first_end),
            };
            if (std::any_of(
                    orientations.begin(),
                    orientations.end(),
                    [guard, &predicate_tolerances](double value) {
                        return std::abs(value)
                            <= guard * predicate_tolerances.area;
                    }
                )) {
                if (shared_edge_vertex_count == 0) {
                    return SurfaceEmbeddingState::CONSTRUCTION_UNDETERMINED;
                }
                continue;
            }
            if (
                std::signbit(orientations[0]) != std::signbit(orientations[1])
                && std::signbit(orientations[2]) != std::signbit(orientations[3])
            ) {
                return SurfaceEmbeddingState::PROVEN_NON_EMBEDDED;
            }
        }
    }

    for (std::size_t direction = 0; direction < 2; ++direction) {
        const TriangleIndices& indices = direction == 0
            ? first.indices
            : second.indices;
        const std::array<Point2, 3>& projected = direction == 0
            ? first_projected
            : second_projected;
        const std::array<Point2, 3>& other_projected = direction == 0
            ? second_projected
            : first_projected;
        for (std::size_t local_index = 0; local_index < 3; ++local_index) {
            if (contains_index(shared, indices[local_index])) {
                continue;
            }
            const PointCycleLocation location = locate_projected_point(
                projected[local_index],
                ArrayView<Point2>(
                    other_projected.data(), other_projected.size()
                ),
                predicate_tolerances,
                tolerances
            );
            if (location == PointCycleLocation::INTERIOR) {
                return SurfaceEmbeddingState::PROVEN_NON_EMBEDDED;
            }
            if (
                location == PointCycleLocation::BOUNDARY
                || location == PointCycleLocation::UNDETERMINED
            ) {
                return SurfaceEmbeddingState::CONSTRUCTION_UNDETERMINED;
            }
        }
    }
    return SurfaceEmbeddingState::EMBEDDED;
}


double point_to_shared_simplex_distance(
    const Point3& point,
    ArrayView<Point3> cycle,
    const std::vector<std::size_t>& shared
) noexcept {
    if (shared.empty()) {
        return std::numeric_limits<double>::infinity();
    }
    if (shared.size() == 1) {
        return norm(subtract(point, cycle[shared[0]]));
    }
    const Point3& start = cycle[shared[0]];
    const Point3& end = cycle[shared[1]];
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


SurfaceEmbeddingState triangle_pair_state(
    const PreparedTriangleGeometry& first,
    const PreparedTriangleGeometry& second,
    ArrayView<Point3> cycle,
    const PredicateTolerances& predicate_tolerances,
    const NumericTolerances& tolerances,
    const std::vector<std::size_t>& shared
) {
    if (
        shared.empty()
        && aabb_stably_separated(
            first.bounds, second.bounds, predicate_tolerances.aabb
        )
    ) {
        return SurfaceEmbeddingState::EMBEDDED;
    }
    const double guard = tolerances.predicate_guard_factor;
    if (
        first.normal_length <= predicate_tolerances.area
        || second.normal_length <= predicate_tolerances.area
    ) {
        return SurfaceEmbeddingState::PROVEN_NON_EMBEDDED;
    }
    if (
        first.normal_length <= guard * predicate_tolerances.area
        || second.normal_length <= guard * predicate_tolerances.area
    ) {
        return SurfaceEmbeddingState::CONSTRUCTION_UNDETERMINED;
    }

    std::vector<double> residuals;
    residuals.reserve(6 - 2 * shared.size());
    for (std::size_t index = 0; index < 3; ++index) {
        if (!contains_index(shared, second.indices[index])) {
            residuals.push_back(std::abs(dot(
                first.normal,
                subtract(second.coordinates[index], first.coordinates[0])
            )));
        }
    }
    for (std::size_t index = 0; index < 3; ++index) {
        if (!contains_index(shared, first.indices[index])) {
            residuals.push_back(std::abs(dot(
                second.normal,
                subtract(first.coordinates[index], second.coordinates[0])
            )));
        }
    }
    if (
        !residuals.empty()
        && std::all_of(
            residuals.begin(),
            residuals.end(),
            [&predicate_tolerances](double value) {
                return value <= predicate_tolerances.volume;
            }
        )
    ) {
        return coplanar_triangle_pair_state(
            first,
            second,
            predicate_tolerances,
            tolerances,
            shared
        );
    }
    if (std::any_of(
            residuals.begin(),
            residuals.end(),
            [guard, &predicate_tolerances](double value) {
                return (
                    predicate_tolerances.volume < value
                    && value <= guard * predicate_tolerances.volume
                );
            }
        )) {
        return SurfaceEmbeddingState::CONSTRUCTION_UNDETERMINED;
    }

    for (std::size_t direction = 0; direction < 2; ++direction) {
        const PreparedTriangleGeometry& triangle = direction == 0
            ? first
            : second;
        const PreparedTriangleGeometry& other = direction == 0
            ? second
            : first;
        for (std::size_t edge_index = 0; edge_index < 3; ++edge_index) {
            const std::size_t edge_start = triangle.indices[edge_index];
            const std::size_t edge_end = triangle.indices[(edge_index + 1) % 3];
            if (
                contains_index(other.indices, edge_start)
                && contains_index(other.indices, edge_end)
            ) {
                continue;
            }
            const TriangleHit hit = segment_triangle_relation(
                triangle.edges[edge_index],
                other,
                predicate_tolerances,
                tolerances
            );
            if (
                hit.kind == TriangleHitKind::DEGENERATE
                || hit.kind == TriangleHitKind::UNDETERMINED
                || hit.kind == TriangleHitKind::COPLANAR
            ) {
                return SurfaceEmbeddingState::CONSTRUCTION_UNDETERMINED;
            }
            if (
                !hit.point.has_value()
                || hit.kind == TriangleHitKind::SEPARATED
                || hit.kind == TriangleHitKind::LINE_EXTENSION_INTERIOR
            ) {
                continue;
            }
            const double distance = point_to_shared_simplex_distance(
                *hit.point, cycle, shared
            );
            if (distance > guard * predicate_tolerances.length) {
                return SurfaceEmbeddingState::PROVEN_NON_EMBEDDED;
            }
            if (distance > predicate_tolerances.length) {
                return SurfaceEmbeddingState::CONSTRUCTION_UNDETERMINED;
            }
        }
    }
    return SurfaceEmbeddingState::EMBEDDED;
}


SurfaceEmbeddingState determine_surface_embedding(
    const PreparedSurfaceGeometry& surface,
    const std::vector<PreparedTriangleGeometry>& triangles,
    ArrayView<Point3> cycle,
    const PredicateTolerances& predicate_tolerances,
    const NumericTolerances& tolerances,
    const SurfaceEnumerationLimits& limits,
    SurfaceCounters& counters
) {
    const double guard = tolerances.predicate_guard_factor;
    for (const std::size_t position : surface.triangle_positions) {
        const double area_measure = triangles[position].normal_length;
        if (area_measure <= predicate_tolerances.area) {
            return SurfaceEmbeddingState::PROVEN_NON_EMBEDDED;
        }
        if (area_measure <= guard * predicate_tolerances.area) {
            return SurfaceEmbeddingState::CONSTRUCTION_UNDETERMINED;
        }
    }

    for (
        std::size_t pair_index = 0;
        pair_index < surface.triangle_pairs.size();
        ++pair_index
    ) {
        if (counters.triangle_pair_tests >= limits.maximum_triangle_pair_tests) {
            counters.triangle_pair_budget_exhausted = true;
            return SurfaceEmbeddingState::CONSTRUCTION_UNDETERMINED;
        }
        ++counters.triangle_pair_tests;
        const IndexPair& pair = surface.triangle_pairs[pair_index];
        const SurfaceEmbeddingState state = triangle_pair_state(
            triangles[surface.triangle_positions[pair[0]]],
            triangles[surface.triangle_positions[pair[1]]],
            cycle,
            predicate_tolerances,
            tolerances,
            surface.shared_simplices[pair_index]
        );
        if (state != SurfaceEmbeddingState::EMBEDDED) {
            return state;
        }
    }
    return SurfaceEmbeddingState::EMBEDDED;
}


}  // namespace


void SurfaceEnumerationLimits::validate() const {
    if (maximum_cycle_vertices < detail::minimum_cycle_vertex_count) {
        throw std::invalid_argument(
            "maximum_cycle_vertices must be at least three"
        );
    }
    if (maximum_surface_count == 0) {
        throw std::invalid_argument(
            "maximum_surface_count must be greater than zero"
        );
    }
    if (maximum_segment_triangle_tests == 0) {
        throw std::invalid_argument(
            "maximum_segment_triangle_tests must be greater than zero"
        );
    }
    if (maximum_triangle_pair_tests == 0) {
        throw std::invalid_argument(
            "maximum_triangle_pair_tests must be greater than zero"
        );
    }
}


PreparedNonplanarSurfaceFamily prepare_nonplanar_surface_family(
    ArrayView<Point3> cycle,
    const NumericTolerances& tolerances,
    const SurfaceEnumerationLimits& limits
) {
    detail::require_cycle(cycle);
    tolerances.validate();
    limits.validate();
    PreparedNonplanarSurfaceFamily prepared{
        std::vector<Point3>(cycle.begin(), cycle.end()),
        tolerances,
        limits,
    };

    if (cycle.size() > limits.maximum_cycle_vertices) {
        add_cause(
            prepared.causes_,
            NonplanarSurfaceCause::INCOMPLETE_SURFACE_FAMILY
        );
        return prepared;
    }
    if (!all_finite(cycle)) {
        prepared.construction_undetermined_count_ = 1;
        add_cause(
            prepared.causes_,
            NonplanarSurfaceCause::SURFACE_CONSTRUCTION
        );
        return prepared;
    }
    const double length_scale = cycle_length_scale(cycle);
    if (
        !std::isfinite(length_scale)
        || length_scale <= tolerances.absolute_length
    ) {
        prepared.construction_undetermined_count_ = 1;
        add_cause(
            prepared.causes_,
            NonplanarSurfaceCause::SURFACE_CONSTRUCTION
        );
        return prepared;
    }
    prepared.predicate_tolerances_ = derive_predicate_tolerances(
        length_scale, tolerances
    );

    const std::vector<SaturatingCount> counts = catalan_counts(cycle.size() - 2);
    const SaturatingCount expected_count = counts[cycle.size() - 2];
    const std::shared_ptr<const CycleTopology> topology_owner = (
        cached_cycle_topology(
            cycle.size(), limits.maximum_surface_count, counts
        )
    );
    const CycleTopology& topology = *topology_owner;
    prepared.unique_triangles_.reserve(topology.unique_triangles.size());
    for (const TriangleIndices& indices : topology.unique_triangles) {
        prepared.unique_triangles_.push_back(prepare_triangle(cycle, indices));
    }
    prepared.enumeration_complete_ = (
        !expected_count.overflowed
        && expected_count.value <= limits.maximum_surface_count
        && topology.surface_count == expected_count.value
    );

    const std::size_t surface_count = topology.surfaces.size();
    SurfaceCounters counters;
    prepared.surfaces_.reserve(surface_count);
    prepared.surface_states_.reserve(surface_count);
    for (
        std::size_t surface_index = 0;
        surface_index < surface_count;
        ++surface_index
    ) {
        if (counters.triangle_pair_budget_exhausted) {
            prepared.enumeration_complete_ = false;
            break;
        }
        const PreparedSurfaceGeometry& surface = topology.surfaces[surface_index];
        const SurfaceEmbeddingState state = determine_surface_embedding(
            surface,
            prepared.unique_triangles_,
            ArrayView<Point3>(prepared.coordinates_),
            *prepared.predicate_tolerances_,
            prepared.tolerances_,
            prepared.limits_,
            counters
        );
        prepared.surfaces_.push_back(surface);
        prepared.surface_states_.push_back(state);
        ++prepared.enumerated_surface_count_;
        if (state == SurfaceEmbeddingState::EMBEDDED) {
            prepared.embedded_surface_indices_.push_back(surface_index);
        } else if (state == SurfaceEmbeddingState::PROVEN_NON_EMBEDDED) {
            ++prepared.proven_non_embedded_surface_count_;
        } else {
            ++prepared.construction_undetermined_count_;
            add_cause(
                prepared.causes_,
                NonplanarSurfaceCause::SURFACE_CONSTRUCTION
            );
            if (counters.triangle_pair_budget_exhausted) {
                prepared.enumeration_complete_ = false;
                break;
            }
        }
    }
    prepared.triangle_pair_tests_used_ = counters.triangle_pair_tests;
    if (!prepared.enumeration_complete_) {
        add_cause(
            prepared.causes_,
            NonplanarSurfaceCause::INCOMPLETE_SURFACE_FAMILY
        );
    }
    return prepared;
}


}  // namespace hotpot::geometry
