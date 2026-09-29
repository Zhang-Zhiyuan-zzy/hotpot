#include "cycle_surface.hpp"
#include "planar_predicates.hpp"
#include "vector_math.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <utility>


namespace hotpot::geometry {
namespace {


using detail::all_finite;
using detail::projected_polygon_simplicity;
using detail::cross;
using detail::diameter;
using detail::dot;
using detail::finite;
using detail::nan_value;
using detail::norm;
using detail::point_distance;
using detail::subtract;


double median_cycle_edge_length(ArrayView<Point3> cycle) {
    std::vector<double> lengths;
    lengths.reserve(cycle.size());
    for (std::size_t index = 0; index < cycle.size(); ++index) {
        lengths.push_back(point_distance(
            cycle[index],
            cycle[(index + 1) % cycle.size()]
        ));
    }
    const std::size_t middle = lengths.size() / 2;
    std::nth_element(lengths.begin(), lengths.begin() + middle, lengths.end());
    if (lengths.size() % 2 == 1) {
        return lengths[middle];
    }
    const double upper = lengths[middle];
    std::nth_element(
        lengths.begin(),
        lengths.begin() + middle - 1,
        lengths.begin() + middle
    );
    return 0.5 * (lengths[middle - 1] + upper);
}


struct OneSidedSvdResult {
    Point3 singular_values;
    std::array<Point3, 3> right_vectors;
};


OneSidedSvdResult one_sided_jacobi_svd(
    std::vector<Point3> matrix
) noexcept {
    double matrix_scale = 0.0;
    for (const Point3& row : matrix) {
        for (const double value : row) {
            matrix_scale = std::max(matrix_scale, std::abs(value));
        }
    }
    if (matrix_scale > 0.0) {
        for (Point3& row : matrix) {
            for (double& value : row) {
                value /= matrix_scale;
            }
        }
    }
    std::array<Point3, 3> vectors = {{
        {1.0, 0.0, 0.0},
        {0.0, 1.0, 0.0},
        {0.0, 0.0, 1.0},
    }};
    for (int sweep = 0; sweep < 64; ++sweep) {
        bool changed = false;
        for (const std::array<std::size_t, 2> pair : {
                 std::array<std::size_t, 2>{0, 1},
                 std::array<std::size_t, 2>{0, 2},
                 std::array<std::size_t, 2>{1, 2},
             }) {
            const std::size_t first = pair[0];
            const std::size_t second = pair[1];
            double first_squared = 0.0;
            double second_squared = 0.0;
            double product = 0.0;
            for (const Point3& row : matrix) {
                first_squared += row[first] * row[first];
                second_squared += row[second] * row[second];
                product += row[first] * row[second];
            }
            const double convergence = (
                std::numeric_limits<double>::epsilon()
                * std::sqrt(first_squared) * std::sqrt(second_squared)
                * std::max<std::size_t>(1, matrix.size())
            );
            if (std::abs(product) <= convergence) {
                continue;
            }
            changed = true;
            const double zeta = (
                (second_squared - first_squared) / (2.0 * product)
            );
            const double tangent = std::copysign(
                1.0 / (std::abs(zeta) + std::hypot(1.0, zeta)),
                zeta
            );
            const double cosine = 1.0 / std::sqrt(1.0 + tangent * tangent);
            const double sine = cosine * tangent;
            for (Point3& row : matrix) {
                const double first_value = row[first];
                const double second_value = row[second];
                row[first] = cosine * first_value - sine * second_value;
                row[second] = sine * first_value + cosine * second_value;
            }
            for (Point3& row : vectors) {
                const double first_value = row[first];
                const double second_value = row[second];
                row[first] = cosine * first_value - sine * second_value;
                row[second] = sine * first_value + cosine * second_value;
            }
        }
        if (!changed) {
            break;
        }
    }

    Point3 unordered_singular_values = {0.0, 0.0, 0.0};
    for (const Point3& row : matrix) {
        for (std::size_t column = 0; column < 3; ++column) {
            unordered_singular_values[column] += row[column] * row[column];
        }
    }
    for (double& value : unordered_singular_values) {
        value = std::sqrt(std::max(0.0, value)) * matrix_scale;
    }
    std::array<std::size_t, 3> order = {0, 1, 2};
    std::sort(
        order.begin(),
        order.end(),
        [&unordered_singular_values](std::size_t first, std::size_t second) {
            return (
                unordered_singular_values[first]
                > unordered_singular_values[second]
            );
        }
    );
    Point3 singular_values{};
    std::array<Point3, 3> sorted_vectors{};
    for (std::size_t rank = 0; rank < 3; ++rank) {
        singular_values[rank] = unordered_singular_values[order[rank]];
        sorted_vectors[rank] = {
            vectors[0][order[rank]],
            vectors[1][order[rank]],
            vectors[2][order[rank]],
        };
    }
    return {singular_values, sorted_vectors};
}


std::pair<Point3, Point3> plane_basis(const Point3& normal) noexcept {
    std::size_t axis_index = 0;
    if (std::abs(normal[1]) < std::abs(normal[axis_index])) {
        axis_index = 1;
    }
    if (std::abs(normal[2]) < std::abs(normal[axis_index])) {
        axis_index = 2;
    }
    Point3 axis = {0.0, 0.0, 0.0};
    axis[axis_index] = 1.0;
    Point3 first = cross(normal, axis);
    const double first_length = norm(first);
    for (double& value : first) {
        value /= first_length;
    }
    return {first, cross(normal, first)};
}


std::vector<Point2> project_to_plane(
    ArrayView<Point3> coordinates,
    const Point3& origin,
    const Point3& normal
) {
    const auto [first, second] = plane_basis(normal);
    std::vector<Point2> projected;
    projected.reserve(coordinates.size());
    for (const Point3& point : coordinates) {
        const Point3 centered = subtract(point, origin);
        projected.push_back({dot(centered, first), dot(centered, second)});
    }
    return projected;
}


PlanarityMeasurement nan_planarity(PlanarityKind kind) noexcept {
    const double nan = nan_value();
    return {
        kind,
        {nan, nan, nan},
        std::nullopt,
        {nan, nan, nan},
        nan,
        nan,
        nan,
        nan,
    };
}


}  // namespace


PredicateTolerances derive_predicate_tolerances(
    double length_scale,
    const NumericTolerances& tolerances
) noexcept {
    const double length = tolerances.effective_length(length_scale);
    return {
        length_scale,
        length,
        tolerances.parameter + length / length_scale,
        length * length_scale,
        length * length_scale * length_scale,
        tolerances.aabb_padding_factor * length,
        tolerances.intersection_merge_factor * length,
    };
}


double cycle_length_scale(ArrayView<Point3> cycle) {
    detail::require_cycle(cycle);
    return std::max(diameter(cycle), median_cycle_edge_length(cycle));
}


double segment_cycle_length_scale(
    const Segment3& segment,
    ArrayView<Point3> cycle
) {
    detail::require_cycle(cycle);
    std::vector<Point3> points(cycle.begin(), cycle.end());
    points.push_back(segment.start);
    points.push_back(segment.end);
    return std::max({
        diameter(ArrayView<Point3>(points)),
        point_distance(segment.start, segment.end),
        median_cycle_edge_length(cycle),
    });
}


PlanarityMeasurement measure_planarity(
    ArrayView<Point3> cycle,
    const NumericTolerances& tolerances
) {
    detail::require_cycle(cycle);
    if (!all_finite(cycle)) {
        return nan_planarity(PlanarityKind::UNDETERMINED);
    }
    const double length_scale = cycle_length_scale(cycle);
    if (!std::isfinite(length_scale)) {
        return nan_planarity(PlanarityKind::UNDETERMINED);
    }
    const double length_tolerance = length_scale > 0.0
        ? derive_predicate_tolerances(length_scale, tolerances).length
        : tolerances.absolute_length;

    Point3 centroid = {0.0, 0.0, 0.0};
    for (const Point3& point : cycle) {
        for (std::size_t axis = 0; axis < 3; ++axis) {
            centroid[axis] += point[axis];
        }
    }
    for (double& value : centroid) {
        value /= static_cast<double>(cycle.size());
    }

    std::vector<Point3> centered;
    centered.reserve(cycle.size());
    for (const Point3& point : cycle) {
        centered.push_back(subtract(point, centroid));
    }
    const OneSidedSvdResult svd = one_sided_jacobi_svd(centered);
    const Point3& singular_values = svd.singular_values;

    if (length_scale <= tolerances.absolute_length) {
        return {
            PlanarityKind::DEGENERATE,
            centroid,
            std::nullopt,
            singular_values,
            nan_value(),
            nan_value(),
            length_scale,
            length_tolerance,
        };
    }

    const PredicateTolerances predicate_tolerances = (
        derive_predicate_tolerances(length_scale, tolerances)
    );
    const double guard = tolerances.predicate_guard_factor;
    PlanarityKind kind;
    if (singular_values[0] <= predicate_tolerances.length) {
        kind = PlanarityKind::DEGENERATE;
    } else if (singular_values[0] <= guard * predicate_tolerances.length) {
        kind = PlanarityKind::UNDETERMINED;
    } else {
        const double rank_measure = singular_values[1] / singular_values[0];
        if (rank_measure <= predicate_tolerances.parameter) {
            kind = PlanarityKind::DEGENERATE;
        } else if (rank_measure <= guard * predicate_tolerances.parameter) {
            kind = PlanarityKind::UNDETERMINED;
        } else {
            const Point3 normal = svd.right_vectors[2];
            double maximum_deviation = 0.0;
            double squared_deviation_sum = 0.0;
            for (const Point3& point : centered) {
                const double deviation = std::abs(dot(point, normal));
                maximum_deviation = std::max(maximum_deviation, deviation);
                squared_deviation_sum += deviation * deviation;
            }
            const double rms_deviation = std::sqrt(
                squared_deviation_sum / static_cast<double>(centered.size())
            );
            const double plane_tolerance = (
                tolerances.planarity_factor * predicate_tolerances.length
            );
            if (maximum_deviation <= plane_tolerance) {
                kind = PlanarityKind::PLANAR;
            } else if (maximum_deviation <= guard * plane_tolerance) {
                kind = PlanarityKind::UNDETERMINED;
            } else {
                kind = PlanarityKind::NONPLANAR;
            }
            return {
                kind,
                centroid,
                normal,
                singular_values,
                maximum_deviation,
                rms_deviation,
                length_scale,
                predicate_tolerances.length,
            };
        }
    }
    return {
        kind,
        centroid,
        std::nullopt,
        singular_values,
        nan_value(),
        nan_value(),
        length_scale,
        predicate_tolerances.length,
    };
}


PreparedPlanarCycle prepare_planar_cycle(
    ArrayView<Point3> cycle,
    const NumericTolerances& tolerances
) {
    detail::require_cycle(cycle);
    PreparedPlanarCycle prepared{
        std::vector<Point3>(cycle.begin(), cycle.end()),
        aabb_bounds(cycle),
        measure_planarity(cycle, tolerances),
        tolerances,
        {},
        PolygonSimplicity::UNDETERMINED,
    };
    if (!prepared.has_planar_surface()) {
        return prepared;
    }
    prepared.projection = project_to_plane(
        cycle,
        prepared.planarity.centroid,
        *prepared.planarity.normal
    );
    const PredicateTolerances predicate_tolerances = derive_predicate_tolerances(
        cycle_length_scale(cycle),
        tolerances
    );
    prepared.simplicity = projected_polygon_simplicity(
        ArrayView<Point2>(prepared.projection),
        predicate_tolerances,
        tolerances
    );
    return prepared;
}


Point2 project_point_to_plane(
    const Point3& point,
    const Point3& origin,
    const Point3& normal
) noexcept {
    const auto [first, second] = plane_basis(normal);
    const Point3 centered = subtract(point, origin);
    return {dot(centered, first), dot(centered, second)};
}


PointCycleLocation locate_point_in_planar_cycle(
    const Point3& point,
    const PreparedPlanarCycle& cycle,
    const Point3& plane_origin,
    const Point3& plane_normal
) {
    const ArrayView<Point3> coordinates(cycle.coordinates);
    detail::require_cycle(coordinates);
    const NumericTolerances& tolerances = cycle.tolerances;
    if (
        !finite(point)
        || !finite(plane_origin)
        || !finite(plane_normal)
        || !all_finite(coordinates)
    ) {
        return PointCycleLocation::UNDETERMINED;
    }
    std::vector<Point3> scale_points(coordinates.begin(), coordinates.end());
    scale_points.push_back(point);
    const double length_scale = std::max(
        diameter(ArrayView<Point3>(scale_points)),
        median_cycle_edge_length(coordinates)
    );
    if (length_scale <= tolerances.absolute_length) {
        return PointCycleLocation::UNDETERMINED;
    }
    const PredicateTolerances predicate_tolerances = derive_predicate_tolerances(
        length_scale,
        tolerances
    );
    const std::vector<Point2> polygon = project_to_plane(
        coordinates,
        plane_origin,
        plane_normal
    );
    if (projected_polygon_simplicity(
            ArrayView<Point2>(polygon),
            predicate_tolerances,
            tolerances
        ) != PolygonSimplicity::SIMPLE) {
        return PointCycleLocation::UNDETERMINED;
    }
    const std::array<Point3, 1> point_array = {point};
    const Point2 projected_point = project_to_plane(
        ArrayView<Point3>(point_array.data(), point_array.size()),
        plane_origin,
        plane_normal
    )[0];
    return locate_projected_point(
        projected_point,
        ArrayView<Point2>(polygon),
        predicate_tolerances,
        tolerances
    );
}


}  // namespace hotpot::geometry
