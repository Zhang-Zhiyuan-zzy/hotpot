#include "cycle_surface.hpp"
#include "primitives.hpp"
#include "segment_cycle.hpp"
#include "spatial.hpp"

#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <type_traits>
#include <vector>


namespace geo = hotpot::geometry;


namespace {


constexpr double comparison_tolerance = 1.0e-12;


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


bool approximately_equal(double first, double second) {
    return std::abs(first - second) <= comparison_tolerance;
}


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


template <typename Value>
bool contains(const std::vector<Value>& values, Value target) {
    return std::find(values.begin(), values.end(), target) != values.end();
}


void test_primitives() {
    const geo::Line3 x_axis{{0.0, 0.0, 0.0}, {1.0, 0.0, 0.0}};
    const geo::Line3 y_axis{{0.0, 0.0, 0.0}, {0.0, 1.0, 0.0}};
    const geo::LineRelation line_relation = geo::determine_line_relation(
        x_axis,
        y_axis,
        tolerances
    );
    assert(line_relation.kind == geo::LineRelationKind::INTERSECTING);
    assert(line_relation.distance.has_value());
    assert(approximately_equal(*line_relation.distance, 0.0));
    assert(approximately_equal(
        geo::line_distance(x_axis, y_axis, tolerances),
        0.0
    ));

    const geo::PointSegmentMeasurement point_segment =
        geo::point_segment_measurement(
            {1.0, 1.0, 0.0},
            {{0.0, 0.0, 0.0}, {2.0, 0.0, 0.0}},
            tolerances
        );
    assert(approximately_equal(point_segment.distance, 1.0));
    assert(approximately_equal(point_segment.parameter, 0.5));
    assert(point_segment.closest_point == geo::Point3({1.0, 0.0, 0.0}));
    assert(approximately_equal(
        geo::point_segment_distance(
            {1.0, 1.0, 0.0},
            {{0.0, 0.0, 0.0}, {2.0, 0.0, 0.0}},
            tolerances
        ),
        1.0
    ));

    const geo::NumericTolerances coarse_tolerances{
        1.0,
        0.0,
        1.0e-10,
        64.0,
        4.0,
        1.0,
        1.0e-10,
        4.0,
        4.0,
    };
    const geo::SegmentSegmentMeasurement short_segments =
        geo::segment_segment_measurement(
            {{0.0, 0.0, 0.0}, {0.5, 0.0, 0.0}},
            {{0.5, 0.1, 0.0}, {0.0, 0.1, 0.0}},
            coarse_tolerances
        );
    assert(short_segments.first_segment_degenerate);
    assert(short_segments.second_segment_degenerate);
    assert(approximately_equal(short_segments.distance, 0.1));
    assert(approximately_equal(
        geo::segment_segment_distance(
            {{0.0, 0.0, 0.0}, {0.5, 0.0, 0.0}},
            {{0.5, 0.1, 0.0}, {0.0, 0.1, 0.0}},
            coarse_tolerances
        ),
        0.1
    ));

    const std::array<geo::Point3, 3> points = {{
        {0.0, 0.0, 0.0},
        {3.0, 4.0, 0.0},
        {0.0, 0.0, 2.0},
    }};
    const std::vector<geo::PointPairDistance> distances =
        geo::point_pair_distances(
            geo::ArrayView<geo::Point3>(points.data(), points.size())
        );
    assert(distances.size() == 3);
    assert(distances[0].first_index == 0);
    assert(distances[0].second_index == 1);
    assert(approximately_equal(distances[0].distance, 5.0));
    const std::array<geo::IndexPair, 1> requested_pairs = {{{1, 2}}};
    const std::vector<geo::PointPairDistance> selected_distances =
        geo::point_pair_distances(
            geo::ArrayView<geo::Point3>(points.data(), points.size()),
            geo::ArrayView<geo::IndexPair>(
                requested_pairs.data(),
                requested_pairs.size()
            )
        );
    assert(selected_distances.size() == 1);
    assert(selected_distances[0].first_index == 1);
    assert(selected_distances[0].second_index == 2);
    const std::vector<geo::PointPairDistance> close_pairs =
        geo::find_point_pairs_below_distance(
            geo::ArrayView<geo::Point3>(points.data(), points.size()),
            3.0
        );
    assert(close_pairs.size() == 1);
    assert(close_pairs[0].first_index == 0);
    assert(close_pairs[0].second_index == 2);
}


void test_spatial() {
    const std::array<geo::Point3, 2> points = {{
        {0.0, -1.0, -2.0},
        {3.0, 4.0, 2.0},
    }};
    const geo::Aabb bounds = geo::aabb_bounds(
        geo::ArrayView<geo::Point3>(points.data(), points.size())
    );
    assert(bounds.minimum == geo::Point3({0.0, -1.0, -2.0}));
    assert(bounds.maximum == geo::Point3({3.0, 4.0, 2.0}));

    const geo::Aabb distant{{5.0, 0.0, 0.0}, {6.0, 1.0, 1.0}};
    assert(geo::aabb_stably_separated(bounds, distant, 1.0));
    assert(!geo::aabb_stably_separated(bounds, distant, 2.0));

    const std::array<geo::Aabb, 2> first = {{bounds, bounds}};
    const std::array<geo::Aabb, 2> second = {{distant, bounds}};
    const std::array<double, 2> paddings = {{0.0, 0.0}};
    const std::vector<std::uint8_t> mask = geo::aabb_separation_mask(
        geo::ArrayView<geo::Aabb>(first.data(), first.size()),
        geo::ArrayView<geo::Aabb>(second.data(), second.size()),
        geo::ArrayView<double>(paddings.data(), paddings.size())
    );
    assert(mask == std::vector<std::uint8_t>({1, 0}));

    const std::array<geo::Segment3, 2> segments = {{
        {{0.0, 0.0, 0.0}, {1.0, 1.0, 1.0}},
        {{5.0, 5.0, 5.0}, {6.0, 6.0, 6.0}},
    }};
    const std::vector<std::uint8_t> segment_mask =
        geo::segment_aabb_separation_mask(
            geo::ArrayView<geo::Segment3>(segments.data(), segments.size()),
            bounds,
            geo::ArrayView<double>(paddings.data(), paddings.size())
        );
    assert(segment_mask == std::vector<std::uint8_t>({0, 1}));

    const std::vector<geo::IndexPair> candidates = geo::aabb_candidate_pairs(
        geo::ArrayView<geo::Aabb>(first.data(), first.size()),
        geo::ArrayView<geo::Aabb>(second.data(), second.size()),
        0.0
    );
    assert(candidates.size() == 2);
    assert(candidates[0] == geo::IndexPair({0, 1}));
    assert(candidates[1] == geo::IndexPair({1, 1}));
}


geo::PreparedPlanarCycle prepare_square() {
    const std::array<geo::Point3, 4> square = {{
        {0.0, 0.0, 0.0},
        {2.0, 0.0, 0.0},
        {2.0, 2.0, 0.0},
        {0.0, 2.0, 0.0},
    }};
    return geo::prepare_planar_cycle(
        geo::ArrayView<geo::Point3>(square.data(), square.size()),
        tolerances
    );
}


void test_planar_cycle_relations() {
    const geo::PreparedPlanarCycle square = prepare_square();
    const geo::PlanarityMeasurement planarity = geo::measure_planarity(
        geo::ArrayView<geo::Point3>(
            square.coordinates().data(),
            square.coordinates().size()
        ),
        tolerances
    );
    assert(planarity.kind == geo::PlanarityKind::PLANAR);
    assert(square.planarity().kind == geo::PlanarityKind::PLANAR);
    assert(square.simplicity() == geo::PolygonSimplicity::SIMPLE);
    assert(square.has_simple_planar_surface());
    assert(square.tolerances().absolute_length == tolerances.absolute_length);

    const geo::Point3 origin = {0.0, 0.0, 0.0};
    const geo::Point3 normal = {0.0, 0.0, 1.0};
    assert(
        geo::locate_point_in_planar_cycle(
            {1.0, 1.0, 0.0}, square, origin, normal
        ) == geo::PointCycleLocation::INTERIOR
    );
    assert(
        geo::locate_point_in_planar_cycle(
            {2.0, 1.0, 0.0}, square, origin, normal
        ) == geo::PointCycleLocation::BOUNDARY
    );
    assert(
        geo::locate_point_in_planar_cycle(
            {3.0, 1.0, 0.0}, square, origin, normal
        ) == geo::PointCycleLocation::EXTERIOR
    );

    const geo::Segment3 piercing{{1.0, 1.0, -1.0}, {1.0, 1.0, 1.0}};
    const geo::Segment3 distant{{4.0, 4.0, -1.0}, {4.0, 4.0, 1.0}};
    const std::optional<geo::ClosestCycleEdge> closest =
        geo::closest_cycle_edge(square, piercing);
    assert(closest.has_value());
    assert(closest->edge_index == 0);
    assert(approximately_equal(closest->distance, 1.0));

    const geo::SegmentCycleRelation piercing_relation =
        geo::determine_planar_segment_cycle_relation(piercing, square);
    assert(piercing_relation.state == geo::PiercingState::PIERCES);
    assert(contains(
        piercing_relation.features,
        geo::SegmentCycleFeature::TRANSVERSE_INTERIOR
    ));
    assert(piercing_relation.intersection_points.size() == 1);
    assert(
        piercing_relation.intersection_points[0]
        == geo::Point3({1.0, 1.0, 0.0})
    );
    assert(piercing_relation.surface_evidence.enumeration_complete);
    assert(piercing_relation.surface_evidence.intersecting_surface_count == 1);

    const std::array<geo::Segment3, 2> segments = {{piercing, distant}};
    const geo::ArrayView<geo::Segment3> segment_view(
        segments.data(),
        segments.size()
    );
    const std::vector<geo::SegmentCycleRelation> relations =
        geo::planar_segment_cycle_relations(segment_view, square);
    assert(relations.size() == 2);
    assert(relations[0].state == geo::PiercingState::PIERCES);
    assert(relations[1].state == geo::PiercingState::DOES_NOT_PIERCE);

    const std::vector<geo::SegmentCycleScreening> screenings =
        geo::planar_segment_cycle_screenings(segment_view, square);
    assert(screenings.size() == 2);
    assert(!screenings[0].aabb_separated);
    assert(screenings[0].relation.has_value());
    assert(screenings[1].aabb_separated);
    assert(!screenings[1].relation.has_value());
    assert(screenings[1].state == geo::PiercingState::DOES_NOT_PIERCE);
}


void test_short_cycle_contract() {
    const std::array<geo::Point3, 2> vertices = {{
        {0.0, 0.0, 0.0},
        {1.0, 0.0, 0.0},
    }};
    const geo::ArrayView<geo::Point3> short_cycle(
        vertices.data(),
        vertices.size()
    );
    const geo::Segment3 segment{{0.5, 0.0, -1.0}, {0.5, 0.0, 1.0}};

    assert_invalid_argument([&] { geo::cycle_length_scale(short_cycle); });
    assert_invalid_argument([&] {
        geo::segment_cycle_length_scale(segment, short_cycle);
    });
    assert_invalid_argument([&] {
        geo::measure_planarity(short_cycle, tolerances);
    });
    assert_invalid_argument([&] {
        geo::prepare_planar_cycle(short_cycle, tolerances);
    });

    static_assert(!std::is_default_constructible_v<geo::PreparedPlanarCycle>);
    static_assert(!std::is_constructible_v<
        geo::PreparedPlanarCycle,
        std::vector<geo::Point3>,
        geo::Aabb,
        geo::PlanarityMeasurement,
        geo::NumericTolerances,
        std::vector<geo::Point2>,
        geo::PolygonSimplicity
    >);
    static_assert(!std::is_copy_assignable_v<geo::PreparedPlanarCycle>);
    static_assert(!std::is_move_assignable_v<geo::PreparedPlanarCycle>);
}


}  // namespace


int main() {
    tolerances.validate();
    test_primitives();
    test_spatial();
    test_planar_cycle_relations();
    test_short_cycle_contract();
    return 0;
}
