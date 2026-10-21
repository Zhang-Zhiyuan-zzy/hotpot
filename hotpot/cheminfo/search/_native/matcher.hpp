#pragma once

#include "compatibility.hpp"
#include "graph_data.hpp"

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <vector>


namespace hotpot::search {

struct EarlierEdge {
    Position neighbor;
    EdgeIndex edge;
};

struct QueryPlan {
    std::vector<Position> order;
    std::vector<Position> order_position;
    std::vector<std::int32_t> parent;
    std::vector<std::vector<EarlierEdge>> earlier_edges;
};

class SearchProblem {
public:
    virtual ~SearchProblem() = default;
    virtual const GraphData& query() const noexcept = 0;
    virtual const GraphData& target() const noexcept = 0;
    virtual const QueryPlan& plan() const noexcept = 0;
    virtual const Compatibility& compatibility() const noexcept = 0;
    virtual const std::vector<std::vector<std::uint64_t>>& domains() const noexcept = 0;
};

class MappingCursor {
public:
    explicit MappingCursor(
        std::shared_ptr<const SearchProblem> problem,
        std::optional<std::pair<Position, Position>> anchor = std::nullopt
    );
    MappingCursor(const MappingCursor&) = delete;
    MappingCursor& operator=(const MappingCursor&) = delete;

    std::optional<std::vector<std::int64_t>> next_mapping();

private:
    struct Frame {
        bool initialized = false;
        std::size_t cursor = 0;
        std::size_t end = 0;
    };

    bool domain_contains(Position query, Position target) const noexcept;
    const QueryPlan& plan() const noexcept {
        return anchored_plan_.has_value() ? *anchored_plan_ : problem_->plan();
    }
    void initialize_frame(std::size_t depth);
    std::optional<Position> next_candidate(std::size_t depth);
    bool candidate_matches(Position query, Position target) const;
    bool forward_checks(Position query, Position target) const;
    bool remaining_vertex_has_candidate(Position query) const;
    void assign(Position query, Position target) noexcept;
    void unassign(Position query, Position target) noexcept;
    void backtrack_after_yield() noexcept;
    std::vector<std::int64_t> output_mapping() const;

    std::shared_ptr<const SearchProblem> problem_;
    std::optional<std::pair<Position, Position>> anchor_;
    std::optional<QueryPlan> anchored_plan_;
    std::vector<std::int32_t> query_to_target_;
    std::vector<std::int32_t> target_to_query_;
    std::vector<Frame> frames_;
    std::size_t depth_ = 0;
    bool resume_after_yield_ = false;
    bool empty_mapping_yielded_ = false;
    bool exhausted_ = false;
    std::atomic_flag advancing_ = ATOMIC_FLAG_INIT;
};

QueryPlan build_query_plan(
    const GraphData& query,
    std::optional<Position> preferred_root = std::nullopt
);

}  // namespace hotpot::search
