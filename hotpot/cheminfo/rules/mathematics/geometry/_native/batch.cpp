#include "batch.hpp"

#include <stdexcept>
#include <utility>


namespace hotpot::geometry {
namespace {


void validate_cycle_csr(
    std::size_t coordinate_count,
    ArrayView<std::size_t> cycle_indices,
    ArrayView<std::size_t> cycle_offsets
) {
    if (cycle_offsets.empty()) {
        throw std::invalid_argument(
            "cycle_offsets must contain the initial zero offset"
        );
    }
    if (cycle_offsets[0] != 0) {
        throw std::invalid_argument("cycle_offsets must start at zero");
    }
    if (cycle_offsets[cycle_offsets.size() - 1] != cycle_indices.size()) {
        throw std::invalid_argument(
            "the final cycle offset must equal the cycle-index count"
        );
    }
    for (
        std::size_t cycle_index = 0;
        cycle_index + 1 < cycle_offsets.size();
        ++cycle_index
    ) {
        const std::size_t start = cycle_offsets[cycle_index];
        const std::size_t end = cycle_offsets[cycle_index + 1];
        if (end < start) {
            throw std::invalid_argument(
                "cycle_offsets must be monotonically nondecreasing"
            );
        }
        if (end - start < detail::minimum_cycle_vertex_count) {
            throw std::invalid_argument(
                "each packed cycle requires at least three vertices"
            );
        }
    }
    for (const std::size_t coordinate_index : cycle_indices) {
        if (coordinate_index >= coordinate_count) {
            throw std::invalid_argument(
                "cycle_indices contains an out-of-range coordinate index"
            );
        }
    }
}


void validate_candidate_pairs(
    const PreparedCycleBatch& cycles,
    std::size_t segment_count,
    ArrayView<SegmentCyclePair> candidate_pairs
) {
    for (const SegmentCyclePair& pair : candidate_pairs) {
        if (pair.segment_index >= segment_count) {
            throw std::invalid_argument(
                "candidate_pairs contains an out-of-range segment index"
            );
        }
        if (pair.cycle_index >= cycles.cycle_count()) {
            throw std::invalid_argument(
                "candidate_pairs contains an out-of-range cycle index"
            );
        }
    }
}


bool should_store_relation(DetailLevel detail, PiercingState state) noexcept {
    return (
        detail == DetailLevel::FULL
        || (
            detail == DetailLevel::ACTIONABLE
            && state != PiercingState::DOES_NOT_PIERCE
        )
    );
}


}  // namespace


PreparedCycleBatch::PreparedCycleBatch(
    std::vector<Point3> coordinates,
    std::vector<std::size_t> cycle_indices,
    std::vector<std::size_t> cycle_offsets,
    std::vector<PreparedCycle> cycles,
    std::vector<Aabb> cycle_bounds
) :
    coordinates_(std::move(coordinates)),
    cycle_indices_(std::move(cycle_indices)),
    cycle_offsets_(std::move(cycle_offsets)),
    cycles_(std::move(cycles)),
    cycle_bounds_(std::move(cycle_bounds)) {}


const PreparedCycle& PreparedCycleBatch::cycle(std::size_t index) const {
    return cycles_.at(index);
}


SegmentCycleBatch::SegmentCycleBatch(
    DetailLevel detail,
    std::size_t requested_pair_count
) :
    detail_(detail),
    requested_pair_count_(requested_pair_count) {
    states_.reserve(requested_pair_count);
    aabb_separated_.reserve(requested_pair_count);
    surface_complete_.reserve(requested_pair_count);
    if (detail != DetailLevel::STATE_ONLY) {
        relation_positions_.reserve(requested_pair_count);
        relations_.reserve(requested_pair_count);
    }
}


void SegmentCycleBatch::append_state(PiercingState state) {
    states_.push_back(state);
    if (state == PiercingState::PIERCES) {
        ++piercing_pair_count_;
    } else if (state == PiercingState::DOES_NOT_PIERCE) {
        ++does_not_pierce_pair_count_;
    } else {
        ++undetermined_pair_count_;
    }
}


PreparedCycleBatch prepare_cycles(
    ArrayView<Point3> coordinates,
    ArrayView<std::size_t> cycle_indices,
    ArrayView<std::size_t> cycle_offsets,
    const NumericTolerances& tolerances,
    const SurfaceEnumerationLimits& limits
) {
    validate_cycle_csr(coordinates.size(), cycle_indices, cycle_offsets);
    std::vector<Point3> stored_coordinates;
    stored_coordinates.reserve(coordinates.size());
    for (const Point3& coordinate : coordinates) {
        stored_coordinates.push_back(coordinate);
    }
    std::vector<std::size_t> stored_indices;
    stored_indices.reserve(cycle_indices.size());
    for (const std::size_t index : cycle_indices) {
        stored_indices.push_back(index);
    }
    std::vector<std::size_t> stored_offsets(
        cycle_offsets.begin(),
        cycle_offsets.end()
    );
    std::vector<PreparedCycle> cycles;
    std::vector<Aabb> cycle_bounds;
    const std::size_t cycle_count = cycle_offsets.size() - 1;
    cycles.reserve(cycle_count);
    cycle_bounds.reserve(cycle_count);
    for (std::size_t cycle_index = 0; cycle_index < cycle_count; ++cycle_index) {
        const std::size_t start = cycle_offsets[cycle_index];
        const std::size_t end = cycle_offsets[cycle_index + 1];
        std::vector<Point3> cycle_coordinates;
        cycle_coordinates.reserve(end - start);
        for (std::size_t position = start; position < end; ++position) {
            cycle_coordinates.push_back(coordinates[cycle_indices[position]]);
        }
        PreparedCycle prepared = prepare_cycle(
            ArrayView<Point3>(cycle_coordinates),
            tolerances,
            limits
        );
        cycle_bounds.push_back(prepared.bounds());
        cycles.push_back(std::move(prepared));
    }
    return PreparedCycleBatch(
        std::move(stored_coordinates),
        std::move(stored_indices),
        std::move(stored_offsets),
        std::move(cycles),
        std::move(cycle_bounds)
    );
}


std::vector<SegmentCycleRelation> determine_segment_cycle_relations(
    const PreparedCycleBatch& cycles,
    ArrayView<Segment3> segments,
    ArrayView<SegmentCyclePair> candidate_pairs
) {
    validate_candidate_pairs(cycles, segments.size(), candidate_pairs);
    std::vector<SegmentCycleRelation> relations;
    relations.reserve(candidate_pairs.size());
    for (const SegmentCyclePair& pair : candidate_pairs) {
        relations.push_back(determine_segment_cycle_relation(
            segments[pair.segment_index],
            cycles.cycle(pair.cycle_index)
        ));
    }
    return relations;
}


SegmentCycleBatch screen_segments(
    const PreparedCycleBatch& cycles,
    ArrayView<Segment3> segments,
    ArrayView<SegmentCyclePair> candidate_pairs,
    DetailLevel detail,
    bool stop_after_confirmed
) {
    validate_candidate_pairs(cycles, segments.size(), candidate_pairs);
    SegmentCycleBatch result(detail, candidate_pairs.size());
    for (
        std::size_t pair_position = 0;
        pair_position < candidate_pairs.size();
        ++pair_position
    ) {
        const SegmentCyclePair& pair = candidate_pairs[pair_position];
        const Segment3& segment = segments[pair.segment_index];
        const PreparedCycle& cycle = cycles.cycle(pair.cycle_index);
        SegmentCycleScreening screening = screen_segment_cycle(
            segment,
            cycle,
            detail == DetailLevel::FULL
        );
        result.append_state(screening.state);
        result.aabb_separated_.push_back(
            static_cast<std::uint8_t>(screening.aabb_separated)
        );
        result.surface_complete_.push_back(
            static_cast<std::uint8_t>(screening.surface_complete)
        );
        if (screening.aabb_separated) {
            ++result.aabb_separated_pair_count_;
        } else {
            ++result.exact_pair_count_;
            if (should_store_relation(detail, screening.state)) {
                result.relation_positions_.push_back(pair_position);
                if (screening.relation.has_value()) {
                    result.relations_.push_back(std::move(*screening.relation));
                } else {
                    result.relations_.push_back(
                        determine_segment_cycle_relation(segment, cycle)
                    );
                }
            }
        }
        if (
            stop_after_confirmed
            && screening.state == PiercingState::PIERCES
        ) {
            break;
        }
    }
    return result;
}


}  // namespace hotpot::geometry
