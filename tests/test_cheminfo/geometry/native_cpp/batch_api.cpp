#include "batch.hpp"

#include <array>
#include <cassert>
#include <cstddef>
#include <stdexcept>
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


const geo::SurfaceEnumerationLimits limits{
    16,
    4096,
    100000,
    100000,
};


template <typename Callback>
void assert_invalid_argument(Callback callback) {
    bool raised = false;
    try {
        callback();
    } catch (const std::invalid_argument&) {
        raised = true;
    }
    assert(raised);
}


geo::PreparedCycleBatch prepare_two_squares() {
    const std::array<geo::Point3, 8> coordinates = {{
        {0.0, 0.0, 0.0},
        {2.0, 0.0, 0.0},
        {2.0, 2.0, 0.0},
        {0.0, 2.0, 0.0},
        {10.0, 0.0, 0.0},
        {12.0, 0.0, 0.0},
        {12.0, 2.0, 0.0},
        {10.0, 2.0, 0.0},
    }};
    const std::array<std::size_t, 8> indices = {{0, 1, 2, 3, 4, 5, 6, 7}};
    const std::array<std::size_t, 3> offsets = {{0, 4, 8}};
    return geo::prepare_cycles(
        geo::ArrayView<geo::Point3>(coordinates.data(), coordinates.size()),
        geo::ArrayView<std::size_t>(indices.data(), indices.size()),
        geo::ArrayView<std::size_t>(offsets.data(), offsets.size()),
        tolerances,
        limits
    );
}


void test_prepared_batch_contract() {
    static_assert(!std::is_default_constructible_v<geo::PreparedCycleBatch>);
    static_assert(!std::is_copy_assignable_v<geo::PreparedCycleBatch>);
    static_assert(!std::is_move_assignable_v<geo::PreparedCycleBatch>);
    const geo::PreparedCycleBatch cycles = prepare_two_squares();
    assert(cycles.coordinate_count() == 8);
    assert(cycles.cycle_count() == 2);
    assert(cycles.cycle_indices().size() == 8);
    assert(cycles.cycle_offsets() == std::vector<std::size_t>({0, 4, 8}));
    assert(cycles.cycle_bounds().size() == 2);
    assert(cycles.cycle(0).planarity().kind == geo::PlanarityKind::PLANAR);
    assert(cycles.cycle(1).bounds().minimum[0] == 10.0);

    const std::array<geo::Point3, 0> no_coordinates = {};
    const std::array<std::size_t, 0> no_indices = {};
    const std::array<std::size_t, 1> empty_offsets = {{0}};
    const geo::PreparedCycleBatch empty = geo::prepare_cycles(
        geo::ArrayView<geo::Point3>(
            no_coordinates.data(),
            no_coordinates.size()
        ),
        geo::ArrayView<std::size_t>(no_indices.data(), no_indices.size()),
        geo::ArrayView<std::size_t>(empty_offsets.data(), empty_offsets.size()),
        tolerances,
        limits
    );
    assert(empty.coordinate_count() == 0);
    assert(empty.cycle_count() == 0);
}


void test_detail_levels_and_early_stop() {
    const geo::PreparedCycleBatch cycles = prepare_two_squares();
    const std::array<geo::Segment3, 5> segments = {{
        {{100.0, 100.0, -1.0}, {100.0, 100.0, 1.0}},
        {{1.0, 1.0, -1.0}, {1.0, 1.0, 1.0}},
        {{11.0, 1.0, -1.0}, {11.0, 1.0, 1.0}},
        {{0.5, 0.5, 0.0}, {1.5, 0.5, 0.0}},
        {{1.0, 1.0, 0.0}, {1.0, 1.0, 0.0}},
    }};
    const std::array<geo::SegmentCyclePair, 7> pairs = {{
        {0, 0},
        {1, 0},
        {1, 1},
        {2, 1},
        {2, 0},
        {3, 0},
        {4, 0},
    }};
    const geo::ArrayView<geo::Segment3> segment_view(
        segments.data(),
        segments.size()
    );
    const geo::ArrayView<geo::SegmentCyclePair> pair_view(
        pairs.data(),
        pairs.size()
    );

    const geo::SegmentCycleBatch state_only = geo::screen_segments(
        cycles,
        segment_view,
        pair_view,
        geo::DetailLevel::STATE_ONLY
    );
    assert(state_only.requested_pair_count() == 7);
    assert(state_only.evaluated_pair_count() == 7);
    assert(state_only.aabb_separated_pair_count() == 3);
    assert(state_only.exact_pair_count() == 4);
    assert(state_only.piercing_pair_count() == 2);
    assert(state_only.does_not_pierce_pair_count() == 4);
    assert(state_only.undetermined_pair_count() == 1);
    assert(state_only.scan_complete());
    assert(state_only.relations().empty());
    assert(state_only.relation_positions().empty());

    const geo::SegmentCycleBatch actionable = geo::screen_segments(
        cycles,
        segment_view,
        pair_view,
        geo::DetailLevel::ACTIONABLE
    );
    assert(actionable.relation_positions() == std::vector<std::size_t>({1, 3, 6}));
    assert(actionable.relations().size() == 3);
    assert(actionable.relations()[0].state == geo::PiercingState::PIERCES);
    assert(actionable.relations()[2].state == geo::PiercingState::UNDETERMINED);

    const geo::SegmentCycleBatch full = geo::screen_segments(
        cycles,
        segment_view,
        pair_view,
        geo::DetailLevel::FULL
    );
    assert(full.relation_positions() == std::vector<std::size_t>({1, 3, 5, 6}));
    assert(full.relations().size() == full.exact_pair_count());

    const geo::SegmentCycleBatch stopped = geo::screen_segments(
        cycles,
        segment_view,
        pair_view,
        geo::DetailLevel::STATE_ONLY,
        true
    );
    assert(stopped.requested_pair_count() == 7);
    assert(stopped.evaluated_pair_count() == 2);
    assert(!stopped.scan_complete());
    assert(stopped.states()[1] == geo::PiercingState::PIERCES);

    const std::array<geo::SegmentCyclePair, 3> final_piercing_pairs = {{
        {0, 0},
        {3, 0},
        {1, 0},
    }};
    const geo::SegmentCycleBatch stopped_at_end = geo::screen_segments(
        cycles,
        segment_view,
        geo::ArrayView<geo::SegmentCyclePair>(
            final_piercing_pairs.data(),
            final_piercing_pairs.size()
        ),
        geo::DetailLevel::STATE_ONLY,
        true
    );
    assert(stopped_at_end.evaluated_pair_count() == 3);
    assert(stopped_at_end.scan_complete());

    const std::vector<geo::SegmentCycleRelation> dense =
        geo::determine_segment_cycle_relations(cycles, segment_view, pair_view);
    assert(dense.size() == pairs.size());
    for (std::size_t position = 0; position < dense.size(); ++position) {
        assert(dense[position].state == state_only.states()[position]);
    }
}


void test_validation_precedes_scanning() {
    const std::array<geo::Point3, 4> coordinates = {{
        {0.0, 0.0, 0.0},
        {1.0, 0.0, 0.0},
        {1.0, 1.0, 0.0},
        {0.0, 1.0, 0.0},
    }};
    const std::array<std::size_t, 4> indices = {{0, 1, 2, 3}};
    const std::array<std::size_t, 0> no_offsets = {};
    assert_invalid_argument([&]() {
        geo::prepare_cycles(
            geo::ArrayView<geo::Point3>(coordinates.data(), coordinates.size()),
            geo::ArrayView<std::size_t>(indices.data(), indices.size()),
            geo::ArrayView<std::size_t>(no_offsets.data(), no_offsets.size()),
            tolerances,
            limits
        );
    });
    const std::array<std::size_t, 2> bad_start = {{1, 4}};
    assert_invalid_argument([&]() {
        geo::prepare_cycles(
            geo::ArrayView<geo::Point3>(coordinates.data(), coordinates.size()),
            geo::ArrayView<std::size_t>(indices.data(), indices.size()),
            geo::ArrayView<std::size_t>(bad_start.data(), bad_start.size()),
            tolerances,
            limits
        );
    });
    const std::array<std::size_t, 2> bad_end = {{0, 3}};
    assert_invalid_argument([&]() {
        geo::prepare_cycles(
            geo::ArrayView<geo::Point3>(coordinates.data(), coordinates.size()),
            geo::ArrayView<std::size_t>(indices.data(), indices.size()),
            geo::ArrayView<std::size_t>(bad_end.data(), bad_end.size()),
            tolerances,
            limits
        );
    });
    const std::array<std::size_t, 5> duplicate_indices = {{0, 1, 2, 3, 0}};
    const std::array<std::size_t, 3> descending_offsets = {{0, 4, 3}};
    assert_invalid_argument([&]() {
        geo::prepare_cycles(
            geo::ArrayView<geo::Point3>(coordinates.data(), coordinates.size()),
            geo::ArrayView<std::size_t>(
                duplicate_indices.data(),
                duplicate_indices.size()
            ),
            geo::ArrayView<std::size_t>(
                descending_offsets.data(),
                descending_offsets.size()
            ),
            tolerances,
            limits
        );
    });
    const std::array<std::size_t, 3> too_short_indices = {{0, 1, 2}};
    const std::array<std::size_t, 3> too_short_offsets = {{0, 2, 3}};
    assert_invalid_argument([&]() {
        geo::prepare_cycles(
            geo::ArrayView<geo::Point3>(coordinates.data(), coordinates.size()),
            geo::ArrayView<std::size_t>(
                too_short_indices.data(),
                too_short_indices.size()
            ),
            geo::ArrayView<std::size_t>(
                too_short_offsets.data(),
                too_short_offsets.size()
            ),
            tolerances,
            limits
        );
    });
    const std::array<std::size_t, 4> bad_indices = {{0, 1, 2, 4}};
    const std::array<std::size_t, 2> offsets = {{0, 4}};
    assert_invalid_argument([&]() {
        geo::prepare_cycles(
            geo::ArrayView<geo::Point3>(coordinates.data(), coordinates.size()),
            geo::ArrayView<std::size_t>(bad_indices.data(), bad_indices.size()),
            geo::ArrayView<std::size_t>(offsets.data(), offsets.size()),
            tolerances,
            limits
        );
    });

    const geo::PreparedCycleBatch cycles = prepare_two_squares();
    const std::array<geo::Segment3, 1> segments = {{
        {{1.0, 1.0, -1.0}, {1.0, 1.0, 1.0}},
    }};
    const std::array<geo::SegmentCyclePair, 2> bad_pairs = {{{0, 0}, {1, 0}}};
    assert_invalid_argument([&]() {
        geo::screen_segments(
            cycles,
            geo::ArrayView<geo::Segment3>(segments.data(), segments.size()),
            geo::ArrayView<geo::SegmentCyclePair>(
                bad_pairs.data(),
                bad_pairs.size()
            ),
            geo::DetailLevel::FULL
        );
    });
}


}  // namespace


int main() {
    test_prepared_batch_contract();
    test_detail_levels_and_early_stop();
    test_validation_precedes_scanning();
    return 0;
}
