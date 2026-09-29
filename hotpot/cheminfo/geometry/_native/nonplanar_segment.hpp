#pragma once

#include "nonplanar_surface.hpp"
#include "segment_cycle.hpp"

#include <vector>


namespace hotpot::geometry {


SegmentCycleRelation determine_nonplanar_segment_cycle_relation(
    const Segment3& segment,
    const PreparedNonplanarSurfaceFamily& family
);


std::vector<SegmentCycleRelation> nonplanar_segment_cycle_relations(
    ArrayView<Segment3> segments,
    const PreparedNonplanarSurfaceFamily& family
);


std::vector<SegmentCycleScreening> nonplanar_segment_cycle_screenings(
    ArrayView<Segment3> segments,
    const PreparedNonplanarSurfaceFamily& family
);


}  // namespace hotpot::geometry
