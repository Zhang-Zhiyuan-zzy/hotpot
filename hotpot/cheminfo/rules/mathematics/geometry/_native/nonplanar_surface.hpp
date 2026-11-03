#pragma once

#include "cycle_surface.hpp"
#include "defaults.hpp"
#include "spatial.hpp"
#include "tolerances.hpp"
#include "types.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <stdexcept>
#include <utility>
#include <vector>


namespace hotpot::geometry {


using TriangleIndices = std::array<std::size_t, 3>;
using EdgeIndices = std::array<std::size_t, 2>;


enum class SurfaceEmbeddingState : std::uint8_t {
    EMBEDDED,
    PROVEN_NON_EMBEDDED,
    CONSTRUCTION_UNDETERMINED,
};


enum class NonplanarSurfaceCause : std::uint8_t {
    INCOMPLETE_SURFACE_FAMILY,
    SURFACE_CONSTRUCTION,
};


struct SurfaceEnumerationLimits {
    std::size_t maximum_cycle_vertices;
    std::size_t maximum_surface_count;
    std::size_t maximum_segment_triangle_tests;
    std::size_t maximum_triangle_pair_tests;

    void validate() const {
        if (maximum_cycle_vertices < 3) {
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
};


inline SurfaceEnumerationLimits default_surface_enumeration_limits() noexcept {
    return {
        detail::default_maximum_surface_cycle_vertices,
        detail::default_maximum_surface_count,
        detail::default_maximum_segment_triangle_tests,
        detail::default_maximum_triangle_pair_tests,
    };
}


struct PreparedTriangleGeometry {
    TriangleIndices indices;
    std::array<Point3, 3> coordinates;
    Point3 normal;
    double normal_length;
    Aabb bounds;
    std::array<Segment3, 3> edges;
};


struct PreparedSurfaceGeometry {
    std::vector<std::size_t> triangle_positions;
    std::vector<EdgeIndices> internal_edges;
    std::vector<IndexPair> triangle_pairs;
    std::vector<std::vector<std::size_t>> shared_simplices;
};


class PreparedNonplanarSurfaceFamily {
public:
    PreparedNonplanarSurfaceFamily(
        const PreparedNonplanarSurfaceFamily&
    ) = default;
    PreparedNonplanarSurfaceFamily(PreparedNonplanarSurfaceFamily&&) = default;
    PreparedNonplanarSurfaceFamily& operator=(
        const PreparedNonplanarSurfaceFamily&
    ) = delete;
    PreparedNonplanarSurfaceFamily& operator=(
        PreparedNonplanarSurfaceFamily&&
    ) = delete;

    const std::vector<Point3>& coordinates() const noexcept {
        return coordinates_;
    }

    const NumericTolerances& tolerances() const noexcept {
        return tolerances_;
    }

    const SurfaceEnumerationLimits& limits() const noexcept {
        return limits_;
    }

    const std::optional<PredicateTolerances>& predicate_tolerances(
    ) const noexcept {
        return predicate_tolerances_;
    }

    bool enumeration_complete() const noexcept {
        return enumeration_complete_;
    }

    std::size_t enumerated_surface_count() const noexcept {
        return enumerated_surface_count_;
    }

    std::size_t embedded_surface_count() const noexcept {
        return embedded_surface_indices_.size();
    }

    std::size_t proven_non_embedded_surface_count() const noexcept {
        return proven_non_embedded_surface_count_;
    }

    std::size_t construction_undetermined_count() const noexcept {
        return construction_undetermined_count_;
    }

    std::size_t triangle_pair_tests_used() const noexcept {
        return triangle_pair_tests_used_;
    }

    const std::vector<NonplanarSurfaceCause>& causes() const noexcept {
        return causes_;
    }

    const std::vector<PreparedTriangleGeometry>& unique_triangles(
    ) const noexcept {
        return unique_triangles_;
    }

    const std::vector<PreparedSurfaceGeometry>& surfaces() const noexcept {
        return surfaces_;
    }

    const std::vector<SurfaceEmbeddingState>& surface_states() const noexcept {
        return surface_states_;
    }

    const std::vector<std::size_t>& embedded_surface_indices() const noexcept {
        return embedded_surface_indices_;
    }

private:
    PreparedNonplanarSurfaceFamily(
        std::vector<Point3> coordinates,
        NumericTolerances tolerances,
        SurfaceEnumerationLimits limits
    ) :
        coordinates_(std::move(coordinates)),
        tolerances_(tolerances),
        limits_(limits) {}

    std::vector<Point3> coordinates_;
    NumericTolerances tolerances_;
    SurfaceEnumerationLimits limits_;
    std::optional<PredicateTolerances> predicate_tolerances_;
    bool enumeration_complete_ = false;
    std::size_t enumerated_surface_count_ = 0;
    std::size_t proven_non_embedded_surface_count_ = 0;
    std::size_t construction_undetermined_count_ = 0;
    std::size_t triangle_pair_tests_used_ = 0;
    std::vector<NonplanarSurfaceCause> causes_;
    std::vector<PreparedTriangleGeometry> unique_triangles_;
    std::vector<PreparedSurfaceGeometry> surfaces_;
    std::vector<SurfaceEmbeddingState> surface_states_;
    std::vector<std::size_t> embedded_surface_indices_;

    friend PreparedNonplanarSurfaceFamily prepare_nonplanar_surface_family(
        ArrayView<Point3> cycle,
        const NumericTolerances& tolerances,
        const SurfaceEnumerationLimits& limits
    );
};


PreparedNonplanarSurfaceFamily prepare_nonplanar_surface_family(
    ArrayView<Point3> cycle,
    const NumericTolerances& tolerances,
    const SurfaceEnumerationLimits& limits
);


}  // namespace hotpot::geometry
