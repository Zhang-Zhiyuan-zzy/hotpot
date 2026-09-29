#include "nonplanar_segment.hpp"

#include <algorithm>
#include <array>
#include <cassert>
#include <type_traits>
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


const geo::SurfaceEnumerationLimits limits{8, 132, 792, 1980};


template <typename Value>
bool contains(const std::vector<Value>& values, Value target) {
    return std::find(values.begin(), values.end(), target) != values.end();
}


geo::PreparedNonplanarSurfaceFamily prepare_warped_square(
    const geo::SurfaceEnumerationLimits& selected_limits = limits
) {
    const std::array<geo::Point3, 4> coordinates = {{
        {0.0, 0.0, 0.0},
        {2.0, 0.0, 0.0},
        {2.0, 2.0, 0.4},
        {0.0, 2.0, 0.0},
    }};
    return geo::prepare_nonplanar_surface_family(
        geo::ArrayView<geo::Point3>(coordinates.data(), coordinates.size()),
        tolerances,
        selected_limits
    );
}


void test_scalar_relation() {
    const geo::PreparedNonplanarSurfaceFamily family = prepare_warped_square();
    const geo::Segment3 segment{{0.6, 0.8, -1.0}, {0.6, 0.8, 1.0}};
    const geo::SegmentCycleRelation relation =
        geo::determine_nonplanar_segment_cycle_relation(segment, family);

    assert(relation.state == geo::PiercingState::PIERCES);
    assert(contains(
        relation.features,
        geo::SegmentCycleFeature::TRANSVERSE_INTERIOR
    ));
    assert(relation.indeterminacy_causes.empty());
    assert(
        relation.surface_model
        == geo::CycleSurfaceModel::VERTEX_TRIANGULATION_FAMILY
    );
    assert(relation.intersection_points.size() == 2);
    assert(relation.closest_boundary_edge.has_value());
    assert(relation.closest_boundary_edge->edge_index == 3);
    assert(relation.surface_evidence.enumeration_complete);
    assert(relation.surface_evidence.enumerated_surface_count == 2);
    assert(relation.surface_evidence.embedded_surface_count == 2);
    assert(relation.surface_evidence.intersecting_surface_count == 2);
    assert(relation.surface_evidence.non_piercing_surface_count == 0);
    assert(relation.surface_evidence.evaluation_undetermined_count == 0);
    assert(relation.surface_evidence.segment_triangle_tests_used == 4);
    assert(relation.surface_evidence.triangle_pair_tests_used == 2);
}


void test_batch_relations_and_screenings() {
    const geo::PreparedNonplanarSurfaceFamily family = prepare_warped_square();
    const std::array<geo::Segment3, 3> segments = {{
        {{0.6, 0.8, -1.0}, {0.6, 0.8, 1.0}},
        {{0.8, 0.8, -1.0}, {0.8, 0.8, 1.0}},
        {{10.0, 10.0, 4.0}, {11.0, 10.0, 4.0}},
    }};
    const geo::ArrayView<geo::Segment3> segment_view(
        segments.data(),
        segments.size()
    );
    const std::vector<geo::SegmentCycleRelation> relations =
        geo::nonplanar_segment_cycle_relations(segment_view, family);
    assert(relations.size() == 3);
    assert(relations[0].state == geo::PiercingState::PIERCES);
    assert(relations[1].state == geo::PiercingState::UNDETERMINED);
    assert(contains(
        relations[1].indeterminacy_causes,
        geo::SegmentCycleIndeterminacy::NUMERIC_BAND
    ));
    assert(relations[2].state == geo::PiercingState::DOES_NOT_PIERCE);

    const std::vector<geo::SegmentCycleScreening> screenings =
        geo::nonplanar_segment_cycle_screenings(segment_view, family);
    assert(screenings.size() == 3);
    assert(!screenings[0].aabb_separated);
    assert(screenings[0].relation.has_value());
    assert(!screenings[1].aabb_separated);
    assert(screenings[1].relation.has_value());
    assert(screenings[2].aabb_separated);
    assert(!screenings[2].relation.has_value());
    assert(screenings[2].surface_complete);
}


void test_segment_budget_resets_per_query() {
    const geo::PreparedNonplanarSurfaceFamily family = prepare_warped_square(
        {8, 132, 1, 1980}
    );
    const std::array<geo::Segment3, 2> segments = {{
        {{0.6, 0.8, -1.0}, {0.6, 0.8, 1.0}},
        {{0.7, 0.9, -1.0}, {0.7, 0.9, 1.0}},
    }};
    const std::vector<geo::SegmentCycleRelation> relations =
        geo::nonplanar_segment_cycle_relations(
            geo::ArrayView<geo::Segment3>(segments.data(), segments.size()),
            family
        );
    assert(relations.size() == 2);
    for (const geo::SegmentCycleRelation& relation : relations) {
        assert(relation.state == geo::PiercingState::UNDETERMINED);
        assert(contains(
            relation.indeterminacy_causes,
            geo::SegmentCycleIndeterminacy::INCOMPLETE_SURFACE_FAMILY
        ));
        assert(relation.surface_evidence.segment_triangle_tests_used == 1);
        assert(relation.surface_evidence.evaluation_undetermined_count == 2);
    }
}


void test_prepared_family_is_factory_constructed_and_immutable() {
    static_assert(
        !std::is_default_constructible_v<
            geo::PreparedNonplanarSurfaceFamily
        >
    );
    static_assert(!std::is_constructible_v<
        geo::PreparedNonplanarSurfaceFamily,
        std::vector<geo::Point3>,
        geo::NumericTolerances,
        geo::SurfaceEnumerationLimits
    >);
    static_assert(
        !std::is_copy_assignable_v<geo::PreparedNonplanarSurfaceFamily>
    );
    static_assert(
        !std::is_move_assignable_v<geo::PreparedNonplanarSurfaceFamily>
    );
}


}  // namespace


int main() {
    tolerances.validate();
    limits.validate();
    test_scalar_relation();
    test_batch_relations_and_screenings();
    test_segment_budget_resets_per_query();
    test_prepared_family_is_factory_constructed_and_immutable();
    return 0;
}
