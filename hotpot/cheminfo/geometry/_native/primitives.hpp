#pragma once

#include "tolerances.hpp"
#include "types.hpp"

#include <vector>


namespace hotpot::geometry {


LineRelation determine_line_relation(
    const Line3& first,
    const Line3& second,
    const NumericTolerances& tolerances
);


double line_distance(
    const Line3& first,
    const Line3& second,
    const NumericTolerances& tolerances
);


PointSegmentMeasurement point_segment_measurement(
    const Point3& point,
    const Segment3& segment,
    const NumericTolerances& tolerances
);


double point_segment_distance(
    const Point3& point,
    const Segment3& segment,
    const NumericTolerances& tolerances
);


SegmentSegmentMeasurement segment_segment_measurement(
    const Segment3& first,
    const Segment3& second,
    const NumericTolerances& tolerances
);


double segment_segment_distance(
    const Segment3& first,
    const Segment3& second,
    const NumericTolerances& tolerances
);


std::vector<PointPairDistance> point_pair_distances(
    ArrayView<Point3> points
);


std::vector<PointPairDistance> point_pair_distances(
    ArrayView<Point3> points,
    ArrayView<IndexPair> pairs
);


std::vector<PointPairDistance> find_point_pairs_below_distance(
    ArrayView<Point3> points,
    double threshold
);


}  // namespace hotpot::geometry
