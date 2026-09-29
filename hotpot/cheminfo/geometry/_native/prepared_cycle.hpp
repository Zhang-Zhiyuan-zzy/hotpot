#pragma once

#include "nonplanar_segment.hpp"
#include "nonplanar_surface.hpp"
#include "segment_cycle.hpp"

#include <optional>
#include <variant>
#include <vector>


namespace hotpot::geometry {


class PreparedCycle {
public:
    PreparedCycle(const PreparedCycle&) = default;
    PreparedCycle(PreparedCycle&&) = default;
    PreparedCycle& operator=(const PreparedCycle&) = delete;
    PreparedCycle& operator=(PreparedCycle&&) = delete;

    const std::vector<Point3>& coordinates() const noexcept;

    const Aabb& bounds() const noexcept {
        return bounds_;
    }

    const PlanarityMeasurement& planarity() const noexcept {
        return planarity_;
    }

    const NumericTolerances& tolerances() const noexcept;

    const SurfaceEnumerationLimits& limits() const noexcept {
        return limits_;
    }

    bool uses_nonplanar_surface_family() const noexcept;

private:
    using Storage = std::variant<
        PreparedPlanarCycle,
        PreparedNonplanarSurfaceFamily
    >;

    PreparedCycle(
        PreparedPlanarCycle planar_cycle,
        SurfaceEnumerationLimits limits
    );

    PreparedCycle(
        PlanarityMeasurement planarity,
        Aabb bounds,
        SurfaceEnumerationLimits limits,
        PreparedNonplanarSurfaceFamily nonplanar_family
    );

    const PreparedPlanarCycle& planar_cycle() const;
    const PreparedNonplanarSurfaceFamily& nonplanar_family() const;

    Aabb bounds_;
    PlanarityMeasurement planarity_;
    SurfaceEnumerationLimits limits_;
    Storage storage_;

    friend PreparedCycle prepare_cycle(
        ArrayView<Point3> cycle,
        const NumericTolerances& tolerances,
        const SurfaceEnumerationLimits& limits
    );
    friend std::optional<ClosestCycleEdge> closest_cycle_edge(
        const PreparedCycle& cycle,
        const Segment3& segment
    );
    friend SegmentCycleRelation determine_segment_cycle_relation(
        const Segment3& segment,
        const PreparedCycle& cycle
    );
    friend std::vector<SegmentCycleRelation> segment_cycle_relations(
        ArrayView<Segment3> segments,
        const PreparedCycle& cycle
    );
    friend std::vector<SegmentCycleScreening> segment_cycle_screenings(
        ArrayView<Segment3> segments,
        const PreparedCycle& cycle
    );
};


PreparedCycle prepare_cycle(
    ArrayView<Point3> cycle,
    const NumericTolerances& tolerances,
    const SurfaceEnumerationLimits& limits
);


std::optional<ClosestCycleEdge> closest_cycle_edge(
    const PreparedCycle& cycle,
    const Segment3& segment
);


SegmentCycleRelation determine_segment_cycle_relation(
    const Segment3& segment,
    const PreparedCycle& cycle
);


std::vector<SegmentCycleRelation> segment_cycle_relations(
    ArrayView<Segment3> segments,
    const PreparedCycle& cycle
);


std::vector<SegmentCycleScreening> segment_cycle_screenings(
    ArrayView<Segment3> segments,
    const PreparedCycle& cycle
);


}  // namespace hotpot::geometry
