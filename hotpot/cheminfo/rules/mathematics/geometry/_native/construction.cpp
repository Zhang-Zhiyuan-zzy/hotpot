#include "construction.hpp"

#include "vector_math.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>


namespace hotpot::geometry::detail {
namespace {


constexpr double pi = 3.141592653589793238462643383279502884;
constexpr double radians_to_degrees = 180.0 / pi;


bool solve_three_by_three(
    std::array<std::array<double, 3>, 3> matrix,
    Point3 rhs,
    Point3& solution,
    double pivot_tolerance
) {
    for (std::size_t pivot = 0; pivot < 3; ++pivot) {
        std::size_t best = pivot;
        for (std::size_t row = pivot + 1; row < 3; ++row) {
            if (std::abs(matrix[row][pivot]) > std::abs(matrix[best][pivot])) {
                best = row;
            }
        }
        if (std::abs(matrix[best][pivot]) <= pivot_tolerance) {
            return false;
        }
        if (best != pivot) {
            std::swap(matrix[best], matrix[pivot]);
            std::swap(rhs[best], rhs[pivot]);
        }
        const double diagonal = matrix[pivot][pivot];
        for (std::size_t column = pivot; column < 3; ++column) {
            matrix[pivot][column] /= diagonal;
        }
        rhs[pivot] /= diagonal;
        for (std::size_t row = 0; row < 3; ++row) {
            if (row == pivot) {
                continue;
            }
            const double factor = matrix[row][pivot];
            for (std::size_t column = pivot; column < 3; ++column) {
                matrix[row][column] -= factor * matrix[pivot][column];
            }
            rhs[row] -= factor * rhs[pivot];
        }
    }
    solution = rhs;
    return true;
}


}  // namespace


SphereShellRelation measure_sphere_pair(
    const Point3& first_center,
    double first_radius,
    const Point3& second_center,
    double second_radius,
    const NumericTolerances& tolerances
) {
    const double center_distance = point_distance(first_center, second_center);
    const double radius_sum = first_radius + second_radius;
    const double radius_difference = std::abs(first_radius - second_radius);
    const double length_tolerance = tolerances.effective_length(std::max({
        center_distance,
        radius_sum,
        radius_difference,
    }));
    return {
        center_distance,
        radius_sum,
        radius_difference,
        center_distance + length_tolerance >= radius_difference
            && center_distance <= radius_sum + length_tolerance,
    };
}


std::optional<double> angle_at_vertex(
    const Point3& first,
    const Point3& vertex,
    const Point3& second,
    const NumericTolerances& tolerances
) {
    const Point3 first_vector = subtract(first, vertex);
    const Point3 second_vector = subtract(second, vertex);
    const double first_length = norm(first_vector);
    const double second_length = norm(second_vector);
    const double length_tolerance = tolerances.effective_length(
        std::max(first_length, second_length)
    );
    if (first_length <= length_tolerance || second_length <= length_tolerance) {
        return std::nullopt;
    }
    const double cosine = std::clamp(
        dot(first_vector, second_vector) / (first_length * second_length),
        -1.0,
        1.0
    );
    return std::acos(cosine) * radians_to_degrees;
}


std::optional<SphereIntersectionCircle> sphere_intersection_circle(
    const Point3& first_center,
    double first_radius,
    const Point3& second_center,
    double second_radius,
    const NumericTolerances& tolerances
) {
    const Point3 axis = subtract(second_center, first_center);
    const double separation = norm(axis);
    const double length_tolerance = tolerances.effective_length(std::max({
        separation,
        first_radius,
        second_radius,
    }));
    const auto relation = measure_sphere_pair(
        first_center,
        first_radius,
        second_center,
        second_radius,
        tolerances
    );
    if (separation <= length_tolerance || !relation.intersects) {
        return std::nullopt;
    }
    const Point3 unit_axis{
        axis[0] / separation,
        axis[1] / separation,
        axis[2] / separation,
    };
    const double axial = (
        first_radius * first_radius
        - second_radius * second_radius
        + separation * separation
    ) / (2.0 * separation);
    const double radial_squared = std::max(
        0.0,
        first_radius * first_radius - axial * axial
    );
    const Point3 circle_center = add_scaled(first_center, unit_axis, axial);
    const Point3 reference = std::abs(unit_axis[0]) < 0.8
        ? Point3{1.0, 0.0, 0.0}
        : Point3{0.0, 1.0, 0.0};
    const Point3 first_basis = *normalized(cross(unit_axis, reference));
    const Point3 second_basis = cross(unit_axis, first_basis);
    const double radius = std::sqrt(radial_squared);
    return SphereIntersectionCircle{
        circle_center,
        first_basis,
        second_basis,
        radius,
    };
}


Point3 point_on_circle(
    const SphereIntersectionCircle& circle,
    double angle_radians
) noexcept {
    Point3 point = circle.center;
    for (std::size_t coordinate = 0; coordinate < 3; ++coordinate) {
        point[coordinate] += circle.radius * (
            std::cos(angle_radians) * circle.first_basis[coordinate]
            + std::sin(angle_radians) * circle.second_basis[coordinate]
        );
    }
    return point;
}


Point3 linearized_sphere_fit_seed(
    ArrayView<Point3> centers,
    ArrayView<double> radii,
    const Point3& fallback,
    const NumericTolerances& tolerances
) {
    if (centers.size() < 2 || centers.size() != radii.size()) {
        return fallback;
    }
    std::array<std::array<double, 3>, 3> normal{};
    Point3 rhs{};
    double length_scale = 0.0;
    for (std::size_t index = 0; index < centers.size(); ++index) {
        length_scale = std::max(length_scale, radii[index]);
        for (std::size_t other = index + 1; other < centers.size(); ++other) {
            length_scale = std::max(
                length_scale,
                point_distance(centers[index], centers[other])
            );
        }
    }
    const double length_tolerance = tolerances.effective_length(length_scale);
    const double matrix_tolerance = length_tolerance * length_tolerance;
    for (std::size_t index = 1; index < centers.size(); ++index) {
        const Point3 offset = subtract(centers[index], centers[0]);
        const Point3 row = {
            2.0 * offset[0],
            2.0 * offset[1],
            2.0 * offset[2],
        };
        const double value = dot(offset, offset)
            + radii[0] * radii[0] - radii[index] * radii[index];
        for (std::size_t first = 0; first < 3; ++first) {
            rhs[first] += row[first] * value;
            for (std::size_t second = 0; second < 3; ++second) {
                normal[first][second] += row[first] * row[second];
            }
            normal[first][first] += matrix_tolerance;
        }
    }
    Point3 relative_result = subtract(fallback, centers[0]);
    static_cast<void>(solve_three_by_three(
        normal,
        rhs,
        relative_result,
        matrix_tolerance
    ));
    return add_scaled(centers[0], relative_result, 1.0);
}


Point3 fit_point_to_spheres(
    Point3 point,
    ArrayView<Point3> centers,
    ArrayView<double> radii,
    std::size_t maximum_iterations,
    const NumericTolerances& tolerances
) {
    double length_scale = 0.0;
    for (std::size_t index = 0; index < centers.size(); ++index) {
        length_scale = std::max(length_scale, radii[index]);
        length_scale = std::max(
            length_scale,
            point_distance(point, centers[index])
        );
        for (std::size_t other = index + 1; other < centers.size(); ++other) {
            length_scale = std::max(
                length_scale,
                point_distance(centers[index], centers[other])
            );
        }
    }
    const double convergence_tolerance = tolerances.effective_length(
        length_scale
    );
    const double pivot_tolerance = std::max(
        tolerances.parameter,
        tolerances.machine_epsilon_factor
            * std::numeric_limits<double>::epsilon()
    );
    for (std::size_t iteration = 0; iteration < maximum_iterations; ++iteration) {
        std::array<std::array<double, 3>, 3> normal{};
        Point3 rhs{0.0, 0.0, 0.0};
        for (std::size_t index = 0; index < centers.size(); ++index) {
            const Point3 delta = subtract(point, centers[index]);
            const double distance = norm(delta);
            if (distance <= pivot_tolerance) {
                continue;
            }
            const Point3 unit{
                delta[0] / distance,
                delta[1] / distance,
                delta[2] / distance,
            };
            const double residual = distance - radii[index];
            for (std::size_t row = 0; row < 3; ++row) {
                rhs[row] -= unit[row] * residual;
                for (std::size_t column = 0; column < 3; ++column) {
                    normal[row][column] += unit[row] * unit[column];
                }
                normal[row][row] += pivot_tolerance;
            }
        }
        Point3 step{};
        if (!solve_three_by_three(normal, rhs, step, pivot_tolerance)) {
            break;
        }
        point = add_scaled(point, step, 1.0);
        if (norm(step) <= convergence_tolerance) {
            break;
        }
    }
    return point;
}


}  // namespace hotpot::geometry::detail
