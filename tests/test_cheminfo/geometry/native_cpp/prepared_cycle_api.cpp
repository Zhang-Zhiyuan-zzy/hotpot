#include "prepared_cycle.hpp"

#include <cassert>
#include <cmath>
#include <stdexcept>
#include <type_traits>
#include <vector>


namespace geo = hotpot::geometry;


namespace {


geo::NumericTolerances tolerances() {
    return {
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
}


geo::SurfaceEnumerationLimits limits() {
    return {8, 132, 792, 1980};
}


bool same_double(double first, double second) {
    return first == second || (std::isnan(first) && std::isnan(second));
}


void assert_same_edge(
    const std::optional<geo::ClosestCycleEdge>& first,
    const std::optional<geo::ClosestCycleEdge>& second
) {
    assert(first.has_value() == second.has_value());
    if (first.has_value()) {
        assert(first->edge_index == second->edge_index);
        assert(same_double(first->distance, second->distance));
    }
}


void assert_same_evidence(
    const geo::SurfaceFamilyEvidence& first,
    const geo::SurfaceFamilyEvidence& second
) {
    assert(first.enumeration_complete == second.enumeration_complete);
    assert(first.enumerated_surface_count == second.enumerated_surface_count);
    assert(first.embedded_surface_count == second.embedded_surface_count);
    assert(
        first.proven_non_embedded_surface_count
        == second.proven_non_embedded_surface_count
    );
    assert(
        first.construction_undetermined_count
        == second.construction_undetermined_count
    );
    assert(first.intersecting_surface_count == second.intersecting_surface_count);
    assert(first.non_piercing_surface_count == second.non_piercing_surface_count);
    assert(
        first.evaluation_undetermined_count
        == second.evaluation_undetermined_count
    );
    assert(
        first.segment_triangle_tests_used
        == second.segment_triangle_tests_used
    );
    assert(first.triangle_pair_tests_used == second.triangle_pair_tests_used);
}


void assert_same_relation(
    const geo::SegmentCycleRelation& first,
    const geo::SegmentCycleRelation& second
) {
    assert(first.state == second.state);
    assert(first.features == second.features);
    assert(first.indeterminacy_causes == second.indeterminacy_causes);
    assert(first.surface_model == second.surface_model);
    assert(first.intersection_points == second.intersection_points);
    assert_same_edge(first.closest_boundary_edge, second.closest_boundary_edge);
    assert_same_evidence(first.surface_evidence, second.surface_evidence);
}


void test_planar_dispatch() {
    const std::vector<geo::Point3> square = {
        {0.0, 0.0, 0.0},
        {2.0, 0.0, 0.0},
        {2.0, 2.0, 0.0},
        {0.0, 2.0, 0.0},
    };
    const geo::ArrayView<geo::Point3> cycle(square);
    const geo::PreparedCycle prepared = geo::prepare_cycle(
        cycle,
        tolerances(),
        limits()
    );
    const geo::PreparedPlanarCycle planar = geo::prepare_planar_cycle(
        cycle,
        tolerances()
    );
    assert(prepared.planarity().kind == geo::PlanarityKind::PLANAR);
    assert(!prepared.uses_nonplanar_surface_family());
    assert(prepared.coordinates() == square);
    assert(prepared.bounds().minimum == geo::Point3({0.0, 0.0, 0.0}));
    assert(prepared.bounds().maximum == geo::Point3({2.0, 2.0, 0.0}));

    const geo::Segment3 piercing{{1.0, 1.0, -1.0}, {1.0, 1.0, 1.0}};
    assert_same_relation(
        geo::determine_segment_cycle_relation(piercing, prepared),
        geo::determine_planar_segment_cycle_relation(piercing, planar)
    );
    assert_same_edge(
        geo::closest_cycle_edge(prepared, piercing),
        geo::closest_cycle_edge(planar, piercing)
    );

    const std::vector<geo::Segment3> segments = {
        piercing,
        {{10.0, 10.0, 1.0}, {11.0, 10.0, 1.0}},
    };
    const geo::ArrayView<geo::Segment3> segment_view(segments);
    const auto dispatched_relations = geo::segment_cycle_relations(
        segment_view,
        prepared
    );
    const auto planar_relations = geo::planar_segment_cycle_relations(
        segment_view,
        planar
    );
    assert(dispatched_relations.size() == planar_relations.size());
    for (std::size_t index = 0; index < planar_relations.size(); ++index) {
        assert_same_relation(
            dispatched_relations[index],
            planar_relations[index]
        );
    }
    const auto screenings = geo::segment_cycle_screenings(
        segment_view,
        prepared
    );
    assert(screenings.size() == 2);
    assert(screenings[1].aabb_separated);
    assert(!screenings[1].relation.has_value());
}


void test_nonplanar_dispatch() {
    const std::vector<geo::Point3> warped_square = {
        {0.0, 0.0, 0.0},
        {2.0, 0.0, 0.0},
        {2.0, 2.0, 0.4},
        {0.0, 2.0, 0.0},
    };
    const geo::ArrayView<geo::Point3> cycle(warped_square);
    const geo::PreparedCycle prepared = geo::prepare_cycle(
        cycle,
        tolerances(),
        limits()
    );
    const geo::PreparedNonplanarSurfaceFamily family =
        geo::prepare_nonplanar_surface_family(cycle, tolerances(), limits());
    assert(prepared.planarity().kind == geo::PlanarityKind::NONPLANAR);
    assert(prepared.uses_nonplanar_surface_family());

    const geo::Segment3 segment{{0.6, 0.8, -1.0}, {0.6, 0.8, 1.0}};
    assert_same_relation(
        geo::determine_segment_cycle_relation(segment, prepared),
        geo::determine_nonplanar_segment_cycle_relation(segment, family)
    );
    const std::vector<geo::Segment3> segments = {
        segment,
        {{10.0, 10.0, 4.0}, {11.0, 10.0, 4.0}},
    };
    const geo::ArrayView<geo::Segment3> segment_view(segments);
    const auto dispatched = geo::segment_cycle_screenings(
        segment_view,
        prepared
    );
    const auto specialized = geo::nonplanar_segment_cycle_screenings(
        segment_view,
        family
    );
    assert(dispatched.size() == specialized.size());
    for (std::size_t index = 0; index < dispatched.size(); ++index) {
        assert(dispatched[index].state == specialized[index].state);
        assert(
            dispatched[index].aabb_separated
            == specialized[index].aabb_separated
        );
        assert(
            dispatched[index].surface_complete
            == specialized[index].surface_complete
        );
        assert(
            dispatched[index].relation.has_value()
            == specialized[index].relation.has_value()
        );
        if (dispatched[index].relation.has_value()) {
            assert_same_relation(
                *dispatched[index].relation,
                *specialized[index].relation
            );
        }
    }
}


void test_degenerate_and_input_contracts() {
    const std::vector<geo::Point3> collinear = {
        {0.0, 0.0, 0.0},
        {1.0, 0.0, 0.0},
        {2.0, 0.0, 0.0},
    };
    const geo::PreparedCycle prepared = geo::prepare_cycle(
        geo::ArrayView<geo::Point3>(collinear),
        tolerances(),
        limits()
    );
    assert(prepared.planarity().kind == geo::PlanarityKind::DEGENERATE);
    assert(!prepared.uses_nonplanar_surface_family());
    const geo::Segment3 segment{{1.0, 1.0, -1.0}, {1.0, 1.0, 1.0}};
    const auto relation = geo::determine_segment_cycle_relation(
        segment,
        prepared
    );
    assert(relation.state == geo::PiercingState::UNDETERMINED);
    assert(
        relation.indeterminacy_causes
        == std::vector<geo::SegmentCycleIndeterminacy>({
            geo::SegmentCycleIndeterminacy::DEGENERATE_CYCLE,
        })
    );

    const std::vector<geo::Point3> short_cycle = {
        {0.0, 0.0, 0.0},
        {1.0, 0.0, 0.0},
    };
    bool rejected = false;
    try {
        static_cast<void>(geo::prepare_cycle(
            geo::ArrayView<geo::Point3>(short_cycle),
            tolerances(),
            limits()
        ));
    } catch (const std::invalid_argument&) {
        rejected = true;
    }
    assert(rejected);
}


}  // namespace


int main() {
    static_assert(!std::is_default_constructible_v<geo::PreparedCycle>);
    static_assert(std::is_copy_constructible_v<geo::PreparedCycle>);
    static_assert(std::is_move_constructible_v<geo::PreparedCycle>);
    static_assert(!std::is_copy_assignable_v<geo::PreparedCycle>);
    static_assert(!std::is_move_assignable_v<geo::PreparedCycle>);
    test_planar_dispatch();
    test_nonplanar_dispatch();
    test_degenerate_and_input_contracts();
}
