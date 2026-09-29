#pragma once

#include "prepared_cycle.hpp"

#include <cstddef>
#include <cstdint>
#include <vector>


namespace hotpot::geometry {


enum class DetailLevel : std::uint8_t {
    STATE_ONLY,
    ACTIONABLE,
    FULL,
};


struct SegmentCyclePair {
    std::size_t segment_index;
    std::size_t cycle_index;
};


class PreparedCycleBatch {
public:
    PreparedCycleBatch(const PreparedCycleBatch&) = default;
    PreparedCycleBatch(PreparedCycleBatch&&) = default;
    PreparedCycleBatch& operator=(const PreparedCycleBatch&) = delete;
    PreparedCycleBatch& operator=(PreparedCycleBatch&&) = delete;

    std::size_t coordinate_count() const noexcept {
        return coordinates_.size();
    }

    std::size_t cycle_count() const noexcept {
        return cycles_.size();
    }

    const std::vector<Point3>& coordinates() const noexcept {
        return coordinates_;
    }

    const std::vector<std::size_t>& cycle_indices() const noexcept {
        return cycle_indices_;
    }

    const std::vector<std::size_t>& cycle_offsets() const noexcept {
        return cycle_offsets_;
    }

    const std::vector<Aabb>& cycle_bounds() const noexcept {
        return cycle_bounds_;
    }

    const PreparedCycle& cycle(std::size_t index) const;

private:
    PreparedCycleBatch(
        std::vector<Point3> coordinates,
        std::vector<std::size_t> cycle_indices,
        std::vector<std::size_t> cycle_offsets,
        std::vector<PreparedCycle> cycles,
        std::vector<Aabb> cycle_bounds
    );

    std::vector<Point3> coordinates_;
    std::vector<std::size_t> cycle_indices_;
    std::vector<std::size_t> cycle_offsets_;
    std::vector<PreparedCycle> cycles_;
    std::vector<Aabb> cycle_bounds_;

    friend PreparedCycleBatch prepare_cycles(
        ArrayView<Point3> coordinates,
        ArrayView<std::size_t> cycle_indices,
        ArrayView<std::size_t> cycle_offsets,
        const NumericTolerances& tolerances,
        const SurfaceEnumerationLimits& limits
    );
};


class SegmentCycleBatch {
public:
    DetailLevel detail() const noexcept {
        return detail_;
    }

    std::size_t requested_pair_count() const noexcept {
        return requested_pair_count_;
    }

    std::size_t evaluated_pair_count() const noexcept {
        return states_.size();
    }

    std::size_t aabb_separated_pair_count() const noexcept {
        return aabb_separated_pair_count_;
    }

    std::size_t exact_pair_count() const noexcept {
        return exact_pair_count_;
    }

    std::size_t piercing_pair_count() const noexcept {
        return piercing_pair_count_;
    }

    std::size_t does_not_pierce_pair_count() const noexcept {
        return does_not_pierce_pair_count_;
    }

    std::size_t undetermined_pair_count() const noexcept {
        return undetermined_pair_count_;
    }

    bool scan_complete() const noexcept {
        return states_.size() == requested_pair_count_;
    }

    const std::vector<PiercingState>& states() const noexcept {
        return states_;
    }

    const std::vector<std::uint8_t>& aabb_separated() const noexcept {
        return aabb_separated_;
    }

    const std::vector<std::uint8_t>& surface_complete() const noexcept {
        return surface_complete_;
    }

    const std::vector<std::size_t>& relation_positions() const noexcept {
        return relation_positions_;
    }

    const std::vector<SegmentCycleRelation>& relations() const noexcept {
        return relations_;
    }

private:
    explicit SegmentCycleBatch(
        DetailLevel detail,
        std::size_t requested_pair_count
    );

    void append_state(PiercingState state);

    DetailLevel detail_;
    std::size_t requested_pair_count_;
    std::size_t aabb_separated_pair_count_ = 0;
    std::size_t exact_pair_count_ = 0;
    std::size_t piercing_pair_count_ = 0;
    std::size_t does_not_pierce_pair_count_ = 0;
    std::size_t undetermined_pair_count_ = 0;
    std::vector<PiercingState> states_;
    std::vector<std::uint8_t> aabb_separated_;
    std::vector<std::uint8_t> surface_complete_;
    std::vector<std::size_t> relation_positions_;
    std::vector<SegmentCycleRelation> relations_;

    friend SegmentCycleBatch screen_segments(
        const PreparedCycleBatch& cycles,
        ArrayView<Segment3> segments,
        ArrayView<SegmentCyclePair> candidate_pairs,
        DetailLevel detail,
        bool stop_after_confirmed
    );
};


PreparedCycleBatch prepare_cycles(
    ArrayView<Point3> coordinates,
    ArrayView<std::size_t> cycle_indices,
    ArrayView<std::size_t> cycle_offsets,
    const NumericTolerances& tolerances,
    const SurfaceEnumerationLimits& limits
);


std::vector<SegmentCycleRelation> determine_segment_cycle_relations(
    const PreparedCycleBatch& cycles,
    ArrayView<Segment3> segments,
    ArrayView<SegmentCyclePair> candidate_pairs
);


SegmentCycleBatch screen_segments(
    const PreparedCycleBatch& cycles,
    ArrayView<Segment3> segments,
    ArrayView<SegmentCyclePair> candidate_pairs,
    DetailLevel detail,
    bool stop_after_confirmed = false
);


}  // namespace hotpot::geometry
