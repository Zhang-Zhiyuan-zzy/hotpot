#pragma once

#include "spatial.hpp"
#include "tolerances.hpp"
#include "types.hpp"

#include <array>
#include <cstdint>
#include <optional>
#include <stdexcept>
#include <utility>
#include <vector>


namespace hotpot::geometry {


using Point2 = std::array<double, 2>;


namespace detail {


inline constexpr std::size_t minimum_cycle_vertex_count = 3;


inline void require_cycle(ArrayView<Point3> cycle) {
    if (cycle.size() < minimum_cycle_vertex_count) {
        throw std::invalid_argument(
            "a cycle requires at least three vertices"
        );
    }
}


}  // namespace detail


enum class PlanarityKind : std::uint8_t {
    PLANAR,
    NONPLANAR,
    DEGENERATE,
    UNDETERMINED,
};


enum class PolygonSimplicity : std::uint8_t {
    SIMPLE,
    SELF_INTERSECTING,
    UNDETERMINED,
};


enum class PointCycleLocation : std::uint8_t {
    INTERIOR,
    BOUNDARY,
    EXTERIOR,
    UNDETERMINED,
};


struct PredicateTolerances {
    double length_scale;
    double length;
    double parameter;
    double area;
    double volume;
    double aabb;
    double merge;
};


struct PlanarityMeasurement {
    PlanarityKind kind;
    Point3 centroid;
    std::optional<Point3> normal;
    Point3 singular_values;
    double maximum_deviation;
    double rms_deviation;
    double length_scale;
    double length_tolerance;
};


class PreparedPlanarCycle {
public:
    PreparedPlanarCycle(const PreparedPlanarCycle&) = default;
    PreparedPlanarCycle(PreparedPlanarCycle&&) = default;
    PreparedPlanarCycle& operator=(const PreparedPlanarCycle&) = delete;
    PreparedPlanarCycle& operator=(PreparedPlanarCycle&&) = delete;

    const std::vector<Point3>& coordinates() const noexcept {
        return coordinates_;
    }

    const Aabb& bounds() const noexcept {
        return bounds_;
    }

    const PlanarityMeasurement& planarity() const noexcept {
        return planarity_;
    }

    const NumericTolerances& tolerances() const noexcept {
        return tolerances_;
    }

    const std::vector<Point2>& projection() const noexcept {
        return projection_;
    }

    PolygonSimplicity simplicity() const noexcept {
        return simplicity_;
    }

    bool has_planar_surface() const noexcept {
        return planarity_.kind == PlanarityKind::PLANAR;
    }

    bool has_simple_planar_surface() const noexcept {
        return (
            has_planar_surface()
            && simplicity_ == PolygonSimplicity::SIMPLE
        );
    }

private:
    PreparedPlanarCycle(
        std::vector<Point3> coordinates,
        Aabb bounds,
        PlanarityMeasurement planarity,
        NumericTolerances tolerances,
        std::vector<Point2> projection,
        PolygonSimplicity simplicity
    ) :
        coordinates_(std::move(coordinates)),
        bounds_(bounds),
        planarity_(planarity),
        tolerances_(tolerances),
        projection_(std::move(projection)),
        simplicity_(simplicity) {}

    std::vector<Point3> coordinates_;
    Aabb bounds_;
    PlanarityMeasurement planarity_;
    NumericTolerances tolerances_;
    std::vector<Point2> projection_;
    PolygonSimplicity simplicity_;

    friend PreparedPlanarCycle prepare_planar_cycle(
        ArrayView<Point3> cycle,
        const NumericTolerances& tolerances
    );
};


PredicateTolerances derive_predicate_tolerances(
    double length_scale,
    const NumericTolerances& tolerances
) noexcept;


double cycle_length_scale(ArrayView<Point3> cycle);


double segment_cycle_length_scale(
    const Segment3& segment,
    ArrayView<Point3> cycle
);


PlanarityMeasurement measure_planarity(
    ArrayView<Point3> cycle,
    const NumericTolerances& tolerances
);


PreparedPlanarCycle prepare_planar_cycle(
    ArrayView<Point3> cycle,
    const NumericTolerances& tolerances
);


Point2 project_point_to_plane(
    const Point3& point,
    const Point3& origin,
    const Point3& normal
) noexcept;


PointCycleLocation locate_projected_point(
    const Point2& point,
    ArrayView<Point2> polygon,
    const PredicateTolerances& predicate_tolerances,
    const NumericTolerances& tolerances
);


PointCycleLocation locate_point_in_planar_cycle(
    const Point3& point,
    const PreparedPlanarCycle& cycle,
    const Point3& plane_origin,
    const Point3& plane_normal
);


}  // namespace hotpot::geometry
