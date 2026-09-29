#include "prepared_cycle.hpp"

#include <utility>


namespace hotpot::geometry {


PreparedCycle::PreparedCycle(
    PreparedPlanarCycle planar_cycle,
    SurfaceEnumerationLimits limits
) :
    bounds_(planar_cycle.bounds()),
    planarity_(planar_cycle.planarity()),
    limits_(limits),
    storage_(
        std::in_place_type<PreparedPlanarCycle>,
        std::move(planar_cycle)
    ) {}


PreparedCycle::PreparedCycle(
    PlanarityMeasurement planarity,
    Aabb bounds,
    SurfaceEnumerationLimits limits,
    PreparedNonplanarSurfaceFamily nonplanar_family
) :
    bounds_(bounds),
    planarity_(std::move(planarity)),
    limits_(limits),
    storage_(
        std::in_place_type<PreparedNonplanarSurfaceFamily>,
        std::move(nonplanar_family)
    ) {}


const std::vector<Point3>& PreparedCycle::coordinates() const noexcept {
    if (uses_nonplanar_surface_family()) {
        return std::get<PreparedNonplanarSurfaceFamily>(storage_).coordinates();
    }
    return std::get<PreparedPlanarCycle>(storage_).coordinates();
}


const NumericTolerances& PreparedCycle::tolerances() const noexcept {
    if (uses_nonplanar_surface_family()) {
        return std::get<PreparedNonplanarSurfaceFamily>(storage_).tolerances();
    }
    return std::get<PreparedPlanarCycle>(storage_).tolerances();
}


bool PreparedCycle::uses_nonplanar_surface_family() const noexcept {
    return std::holds_alternative<PreparedNonplanarSurfaceFamily>(storage_);
}


const PreparedPlanarCycle& PreparedCycle::planar_cycle() const {
    return std::get<PreparedPlanarCycle>(storage_);
}


const PreparedNonplanarSurfaceFamily& PreparedCycle::nonplanar_family() const {
    return std::get<PreparedNonplanarSurfaceFamily>(storage_);
}


PreparedCycle prepare_cycle(
    ArrayView<Point3> cycle,
    const NumericTolerances& tolerances,
    const SurfaceEnumerationLimits& limits
) {
    detail::require_cycle(cycle);
    PreparedPlanarCycle planar_cycle = prepare_planar_cycle(
        cycle,
        tolerances
    );
    if (planar_cycle.planarity().kind != PlanarityKind::NONPLANAR) {
        return PreparedCycle(std::move(planar_cycle), limits);
    }
    const PlanarityMeasurement planarity = planar_cycle.planarity();
    const Aabb bounds = planar_cycle.bounds();
    return PreparedCycle(
        planarity,
        bounds,
        limits,
        prepare_nonplanar_surface_family(cycle, tolerances, limits)
    );
}


std::optional<ClosestCycleEdge> closest_cycle_edge(
    const PreparedCycle& cycle,
    const Segment3& segment
) {
    if (cycle.uses_nonplanar_surface_family()) {
        return detail::closest_cycle_edge(
            ArrayView<Point3>(cycle.coordinates()),
            segment,
            cycle.tolerances()
        );
    }
    return closest_cycle_edge(cycle.planar_cycle(), segment);
}


SegmentCycleRelation determine_segment_cycle_relation(
    const Segment3& segment,
    const PreparedCycle& cycle
) {
    if (cycle.uses_nonplanar_surface_family()) {
        return determine_nonplanar_segment_cycle_relation(
            segment,
            cycle.nonplanar_family()
        );
    }
    return determine_planar_segment_cycle_relation(
        segment,
        cycle.planar_cycle()
    );
}


std::vector<SegmentCycleRelation> segment_cycle_relations(
    ArrayView<Segment3> segments,
    const PreparedCycle& cycle
) {
    if (cycle.uses_nonplanar_surface_family()) {
        return nonplanar_segment_cycle_relations(
            segments,
            cycle.nonplanar_family()
        );
    }
    return planar_segment_cycle_relations(segments, cycle.planar_cycle());
}


std::vector<SegmentCycleScreening> segment_cycle_screenings(
    ArrayView<Segment3> segments,
    const PreparedCycle& cycle
) {
    if (cycle.uses_nonplanar_surface_family()) {
        return nonplanar_segment_cycle_screenings(
            segments,
            cycle.nonplanar_family()
        );
    }
    return planar_segment_cycle_screenings(segments, cycle.planar_cycle());
}


}  // namespace hotpot::geometry
