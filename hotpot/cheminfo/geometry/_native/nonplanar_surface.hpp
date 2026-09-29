#pragma once

#include "cycle_surface.hpp"
#include "spatial.hpp"
#include "tolerances.hpp"
#include "types.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <optional>
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

    void validate() const;
};


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


struct PreparedNonplanarSurfaceFamily {
    std::vector<Point3> coordinates;
    NumericTolerances tolerances;
    SurfaceEnumerationLimits limits;
    std::optional<PredicateTolerances> predicate_tolerances;
    bool enumeration_complete;
    std::size_t enumerated_surface_count;
    std::size_t proven_non_embedded_surface_count;
    std::size_t construction_undetermined_count;
    std::size_t triangle_pair_tests_used;
    std::vector<NonplanarSurfaceCause> causes;
    std::vector<PreparedTriangleGeometry> unique_triangles;
    std::vector<PreparedSurfaceGeometry> surfaces;
    std::vector<SurfaceEmbeddingState> surface_states;
    std::vector<std::size_t> embedded_surface_indices;

    std::size_t embedded_surface_count() const noexcept {
        return embedded_surface_indices.size();
    }
};


PreparedNonplanarSurfaceFamily prepare_nonplanar_surface_family(
    ArrayView<Point3> cycle,
    const NumericTolerances& tolerances,
    const SurfaceEnumerationLimits& limits
);


}  // namespace hotpot::geometry
