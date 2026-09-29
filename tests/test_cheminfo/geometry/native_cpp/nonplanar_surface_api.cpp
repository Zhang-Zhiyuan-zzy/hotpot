#include "nonplanar_surface.hpp"

#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <limits>
#include <vector>


namespace geo = hotpot::geometry;


namespace {


const geo::NumericTolerances tolerances{
    1.0e-8,
    1.0e-10,
    1.0e-10,
    64.0,
    4.0,
    1.0,
    1.0e-10,
    4.0,
    4.0,
};


geo::SurfaceEnumerationLimits limits(
    std::size_t maximum_cycle_vertices = 8,
    std::size_t maximum_surface_count = 132,
    std::size_t maximum_triangle_pair_tests = 1980
) {
    return {
        maximum_cycle_vertices,
        maximum_surface_count,
        792,
        maximum_triangle_pair_tests,
    };
}


geo::PreparedNonplanarSurfaceFamily prepare(
    const std::vector<geo::Point3>& cycle,
    const geo::SurfaceEnumerationLimits& surface_limits = limits()
) {
    return geo::prepare_nonplanar_surface_family(
        geo::ArrayView<geo::Point3>(cycle),
        tolerances,
        surface_limits
    );
}


std::vector<geo::TriangleIndices> surface_triangles(
    const geo::PreparedNonplanarSurfaceFamily& family,
    std::size_t surface_index
) {
    std::vector<geo::TriangleIndices> triangles;
    for (
        const std::size_t position
        : family.surfaces()[surface_index].triangle_positions
    ) {
        triangles.push_back(family.unique_triangles()[position].indices);
    }
    return triangles;
}


void test_exact_catalan_order() {
    const std::vector<geo::Point3> triangle = {
        {0.0, 0.0, 0.0},
        {1.0, 0.0, 0.0},
        {0.0, 1.0, 0.1},
    };
    const geo::PreparedNonplanarSurfaceFamily triangle_family = prepare(
        triangle
    );
    assert(triangle_family.enumeration_complete());
    assert(triangle_family.enumerated_surface_count() == 1);
    assert(surface_triangles(triangle_family, 0) == (
        std::vector<geo::TriangleIndices>{{0, 1, 2}}
    ));

    const std::vector<geo::Point3> square = {
        {0.0, 0.0, 0.0},
        {2.0, 0.0, 0.0},
        {2.0, 2.0, 0.4},
        {0.0, 2.0, 0.0},
    };
    const geo::PreparedNonplanarSurfaceFamily square_family = prepare(square);
    assert(square_family.enumeration_complete());
    assert(square_family.enumerated_surface_count() == 2);
    assert(surface_triangles(square_family, 0) == (
        std::vector<geo::TriangleIndices>{{1, 2, 3}, {0, 1, 3}}
    ));
    assert(surface_triangles(square_family, 1) == (
        std::vector<geo::TriangleIndices>{{0, 1, 2}, {0, 2, 3}}
    ));

    const std::vector<geo::Point3> pentagon = {
        {0.0, 0.0, 0.0},
        {2.0, 0.0, 0.1},
        {3.0, 1.0, -0.1},
        {1.5, 2.0, 0.2},
        {0.0, 1.0, 0.0},
    };
    const geo::PreparedNonplanarSurfaceFamily pentagon_family = prepare(
        pentagon
    );
    const std::vector<std::vector<geo::TriangleIndices>> expected = {
        {{2, 3, 4}, {1, 2, 4}, {0, 1, 4}},
        {{1, 2, 3}, {1, 3, 4}, {0, 1, 4}},
        {{0, 1, 2}, {2, 3, 4}, {0, 2, 4}},
        {{1, 2, 3}, {0, 1, 3}, {0, 3, 4}},
        {{0, 1, 2}, {0, 2, 3}, {0, 3, 4}},
    };
    assert(pentagon_family.enumeration_complete());
    assert(pentagon_family.enumerated_surface_count() == expected.size());
    for (std::size_t index = 0; index < expected.size(); ++index) {
        assert(surface_triangles(pentagon_family, index) == expected[index]);
    }
}


void test_bounded_prefix_enumeration() {
    std::vector<geo::Point3> cycle;
    constexpr std::size_t vertex_count = 16;
    constexpr double pi = 3.141592653589793238462643383279502884;
    for (std::size_t index = 0; index < vertex_count; ++index) {
        const double angle = 2.0 * pi * static_cast<double>(index)
            / static_cast<double>(vertex_count);
        cycle.push_back({
            std::cos(angle),
            std::sin(angle),
            index % 2 == 0 ? -0.05 : 0.07,
        });
    }
    const geo::PreparedNonplanarSurfaceFamily family = prepare(
        cycle, limits(16, 2, 1000)
    );
    assert(!family.enumeration_complete());
    assert(family.enumerated_surface_count() == 2);
    assert(family.surfaces().size() == 2);
    assert(std::find(
        family.causes().begin(),
        family.causes().end(),
        geo::NonplanarSurfaceCause::INCOMPLETE_SURFACE_FAMILY
    ) != family.causes().end());

    const std::vector<geo::TriangleIndices> first = surface_triangles(
        family, 0
    );
    assert(first.front() == geo::TriangleIndices({13, 14, 15}));
    assert(first.back() == geo::TriangleIndices({0, 1, 15}));
}


void test_embedding_states_and_shared_budget() {
    const std::vector<geo::Point3> warped_square = {
        {0.0, 0.0, 0.0},
        {2.0, 0.0, 0.0},
        {2.0, 2.0, 0.4},
        {0.0, 2.0, 0.0},
    };
    const geo::PreparedNonplanarSurfaceFamily warped = prepare(warped_square);
    assert(warped.enumeration_complete());
    assert(warped.enumerated_surface_count() == 2);
    assert(warped.embedded_surface_count() == 2);
    assert(warped.proven_non_embedded_surface_count() == 0);
    assert(warped.construction_undetermined_count() == 0);
    assert(warped.triangle_pair_tests_used() == 2);
    assert(warped.surface_states() == std::vector<geo::SurfaceEmbeddingState>({
        geo::SurfaceEmbeddingState::EMBEDDED,
        geo::SurfaceEmbeddingState::EMBEDDED,
    }));

    const geo::PreparedNonplanarSurfaceFamily exhausted = prepare(
        warped_square, limits(8, 132, 1)
    );
    assert(!exhausted.enumeration_complete());
    assert(exhausted.enumerated_surface_count() == 2);
    assert(exhausted.embedded_surface_count() == 1);
    assert(exhausted.proven_non_embedded_surface_count() == 0);
    assert(exhausted.construction_undetermined_count() == 1);
    assert(exhausted.triangle_pair_tests_used() == 1);
    assert(exhausted.surface_states() == (
        std::vector<geo::SurfaceEmbeddingState>{
            geo::SurfaceEmbeddingState::EMBEDDED,
            geo::SurfaceEmbeddingState::CONSTRUCTION_UNDETERMINED,
        }
    ));

    const std::vector<geo::Point3> proven_nonembedded = {
        {0.6, -1.2, 1.2},
        {0.3, -0.9, 1.9},
        {-2.5, -0.2, 1.2},
        {0.5, -0.4, -0.6},
        {-2.5, 1.1, -0.5},
    };
    const geo::PreparedNonplanarSurfaceFamily nonembedded = prepare(
        proven_nonembedded
    );
    assert(nonembedded.surface_states() == (
        std::vector<geo::SurfaceEmbeddingState>{
            geo::SurfaceEmbeddingState::EMBEDDED,
            geo::SurfaceEmbeddingState::PROVEN_NON_EMBEDDED,
            geo::SurfaceEmbeddingState::EMBEDDED,
            geo::SurfaceEmbeddingState::PROVEN_NON_EMBEDDED,
            geo::SurfaceEmbeddingState::EMBEDDED,
        }
    ));
    assert(nonembedded.embedded_surface_indices() == (
        std::vector<std::size_t>{0, 2, 4}
    ));
    assert(nonembedded.triangle_pair_tests_used() == 13);

    const std::vector<geo::Point3> uncertain = {
        {0.0, 0.0, 0.0},
        {2.0, 0.0, 0.0},
        {2.0, 2.0, 1.0},
        {2.0, 0.0, 0.0},
        {0.0, 2.0, 0.0},
    };
    const geo::PreparedNonplanarSurfaceFamily construction = prepare(
        uncertain
    );
    assert(construction.surface_states() == (
        std::vector<geo::SurfaceEmbeddingState>{
            geo::SurfaceEmbeddingState::CONSTRUCTION_UNDETERMINED,
            geo::SurfaceEmbeddingState::PROVEN_NON_EMBEDDED,
            geo::SurfaceEmbeddingState::PROVEN_NON_EMBEDDED,
            geo::SurfaceEmbeddingState::PROVEN_NON_EMBEDDED,
            geo::SurfaceEmbeddingState::CONSTRUCTION_UNDETERMINED,
        }
    ));
    assert(construction.triangle_pair_tests_used() == 3);
}


void test_prepared_family_owns_inputs_and_settings() {
    std::vector<geo::Point3> cycle = {
        {0.0, 0.0, 0.0},
        {2.0, 0.0, 0.0},
        {2.0, 2.0, 0.4},
        {0.0, 2.0, 0.0},
    };
    const geo::SurfaceEnumerationLimits surface_limits = limits(9, 17, 23);
    const geo::PreparedNonplanarSurfaceFamily family = prepare(
        cycle, surface_limits
    );
    cycle[0] = {99.0, 99.0, 99.0};

    assert(family.coordinates()[0] == geo::Point3({0.0, 0.0, 0.0}));
    assert(family.tolerances().absolute_length == tolerances.absolute_length);
    assert(family.limits().maximum_cycle_vertices == 9);
    assert(family.limits().maximum_surface_count == 17);
    assert(family.limits().maximum_segment_triangle_tests == 792);
    assert(family.limits().maximum_triangle_pair_tests == 23);
    assert(family.predicate_tolerances().has_value());
}


void test_nonfinite_and_tiny_cycles_fail_closed() {
    const std::vector<geo::Point3> nonfinite = {
        {0.0, 0.0, 0.0},
        {1.0, 0.0, 0.0},
        {0.0, std::numeric_limits<double>::quiet_NaN(), 0.0},
    };
    const geo::PreparedNonplanarSurfaceFamily nonfinite_family = prepare(
        nonfinite
    );
    assert(!nonfinite_family.enumeration_complete());
    assert(nonfinite_family.enumerated_surface_count() == 0);
    assert(nonfinite_family.construction_undetermined_count() == 1);
    assert(nonfinite_family.causes() == (
        std::vector<geo::NonplanarSurfaceCause>{
            geo::NonplanarSurfaceCause::SURFACE_CONSTRUCTION,
        }
    ));

    const std::vector<geo::Point3> tiny(4, {0.0, 0.0, 0.0});
    const geo::PreparedNonplanarSurfaceFamily tiny_family = prepare(tiny);
    assert(!tiny_family.enumeration_complete());
    assert(tiny_family.enumerated_surface_count() == 0);
    assert(tiny_family.construction_undetermined_count() == 1);
    assert(tiny_family.triangle_pair_tests_used() == 0);
}


}  // namespace


int main() {
    test_exact_catalan_order();
    test_bounded_prefix_enumeration();
    test_embedding_states_and_shared_budget();
    test_prepared_family_owns_inputs_and_settings();
    test_nonfinite_and_tiny_cycles_fail_closed();
    return 0;
}
