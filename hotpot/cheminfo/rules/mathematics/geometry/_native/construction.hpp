#pragma once

#include "tolerances.hpp"
#include "types.hpp"

#include <cstddef>
#include <optional>
#include <vector>


namespace hotpot::geometry::detail {


struct SphereShellRelation {
    double center_distance;
    double radius_sum;
    double radius_difference;
    bool intersects;
};


struct SphereIntersectionCircle {
    Point3 center;
    Point3 first_basis;
    Point3 second_basis;
    double radius;
};


SphereShellRelation measure_sphere_pair(
    const Point3& first_center,
    double first_radius,
    const Point3& second_center,
    double second_radius,
    const NumericTolerances& tolerances
);


std::optional<double> angle_at_vertex(
    const Point3& first,
    const Point3& vertex,
    const Point3& second,
    const NumericTolerances& tolerances
);


std::optional<SphereIntersectionCircle> sphere_intersection_circle(
    const Point3& first_center,
    double first_radius,
    const Point3& second_center,
    double second_radius,
    const NumericTolerances& tolerances
);


Point3 point_on_circle(
    const SphereIntersectionCircle& circle,
    double angle_radians
) noexcept;


Point3 linearized_sphere_fit_seed(
    ArrayView<Point3> centers,
    ArrayView<double> radii,
    const Point3& fallback,
    const NumericTolerances& tolerances
);


Point3 fit_point_to_spheres(
    Point3 point,
    ArrayView<Point3> centers,
    ArrayView<double> radii,
    std::size_t maximum_iterations,
    const NumericTolerances& tolerances
);


}  // namespace hotpot::geometry::detail
