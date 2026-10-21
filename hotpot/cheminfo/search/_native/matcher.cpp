#include "matcher.hpp"

#include <algorithm>
#include <limits>
#include <queue>
#include <stdexcept>


namespace hotpot::search {

namespace {

class AdvancingGuard {
public:
    explicit AdvancingGuard(std::atomic_flag& flag) : flag_(flag) {
        if (flag_.test_and_set(std::memory_order_acquire)) {
            throw std::runtime_error("a mapping cursor cannot be advanced concurrently");
        }
    }
    ~AdvancingGuard() { flag_.clear(std::memory_order_release); }

private:
    std::atomic_flag& flag_;
};

bool bit_is_set(
    const std::vector<std::uint64_t>& words,
    const Position index
) noexcept {
    return (words[index / 64] & (std::uint64_t{1} << (index % 64))) != 0;
}

}  // namespace

QueryPlan build_query_plan(const GraphData& query) {
    const std::size_t node_count = query.node_count();
    QueryPlan plan;
    plan.order_position.assign(node_count, 0);
    plan.parent.assign(node_count, -1);
    plan.earlier_edges.resize(node_count);

    std::vector<bool> discovered(node_count, false);
    std::vector<std::vector<Position>> components;
    for (Position start = 0; start < node_count; ++start) {
        if (discovered[start]) {
            continue;
        }
        std::vector<Position> component;
        std::queue<Position> pending;
        discovered[start] = true;
        pending.push(start);
        while (!pending.empty()) {
            const Position node = pending.front();
            pending.pop();
            component.push_back(node);
            for (
                Position adjacency = query.offsets()[node];
                adjacency < query.offsets()[node + 1];
                ++adjacency
            ) {
                const Position neighbor = query.neighbors()[adjacency];
                if (neighbor != node && !discovered[neighbor]) {
                    discovered[neighbor] = true;
                    pending.push(neighbor);
                }
            }
        }
        std::sort(component.begin(), component.end());
        components.push_back(std::move(component));
    }

    std::vector<bool> ordered(node_count, false);
    for (const auto& component : components) {
        Position root = component.front();
        for (const Position candidate : component) {
            if (
                query.degrees()[candidate] > query.degrees()[root]
                || (
                    query.degrees()[candidate] == query.degrees()[root]
                    && query.keys()[candidate] < query.keys()[root]
                )
            ) {
                root = candidate;
            }
        }
        plan.order.push_back(root);
        ordered[root] = true;

        for (std::size_t placed = 1; placed < component.size(); ++placed) {
            bool found = false;
            Position selected = 0;
            Position selected_mapped_neighbors = 0;
            for (const Position candidate : component) {
                if (ordered[candidate]) {
                    continue;
                }
                Position mapped_neighbors = 0;
                for (
                    Position adjacency = query.offsets()[candidate];
                    adjacency < query.offsets()[candidate + 1];
                    ++adjacency
                ) {
                    const Position neighbor = query.neighbors()[adjacency];
                    if (neighbor != candidate && ordered[neighbor]) {
                        ++mapped_neighbors;
                    }
                }
                if (mapped_neighbors == 0) {
                    continue;
                }
                if (
                    !found
                    || mapped_neighbors > selected_mapped_neighbors
                    || (
                        mapped_neighbors == selected_mapped_neighbors
                        && query.degrees()[candidate] > query.degrees()[selected]
                    )
                    || (
                        mapped_neighbors == selected_mapped_neighbors
                        && query.degrees()[candidate] == query.degrees()[selected]
                        && query.keys()[candidate] < query.keys()[selected]
                    )
                ) {
                    found = true;
                    selected = candidate;
                    selected_mapped_neighbors = mapped_neighbors;
                }
            }
            if (!found) {
                throw std::logic_error("connected query component has no frontier");
            }
            plan.order.push_back(selected);
            ordered[selected] = true;
        }
    }

    for (Position order_position = 0; order_position < plan.order.size(); ++order_position) {
        plan.order_position[plan.order[order_position]] = order_position;
    }
    for (const Position query_node : plan.order) {
        std::int32_t parent = -1;
        Position parent_order = 0;
        for (
            Position adjacency = query.offsets()[query_node];
            adjacency < query.offsets()[query_node + 1];
            ++adjacency
        ) {
            const Position neighbor = query.neighbors()[adjacency];
            if (neighbor == query_node) {
                continue;
            }
            if (plan.order_position[neighbor] < plan.order_position[query_node]) {
                plan.earlier_edges[query_node].push_back({
                    neighbor,
                    query.adjacency_edges()[adjacency],
                });
                if (
                    parent < 0
                    || plan.order_position[neighbor] > parent_order
                ) {
                    parent = static_cast<std::int32_t>(neighbor);
                    parent_order = plan.order_position[neighbor];
                }
            }
        }
        plan.parent[query_node] = parent;
    }
    return plan;
}

SyntheticProblem::SyntheticProblem(
    std::vector<std::int64_t> query_keys,
    const std::vector<std::pair<std::int64_t, std::int64_t>>& query_edges,
    std::vector<std::int64_t> target_keys,
    const std::vector<std::pair<std::int64_t, std::int64_t>>& target_edges,
    const std::vector<std::vector<std::uint8_t>>& node_compatible,
    const std::vector<std::vector<std::uint8_t>>& edge_compatible
) : query_(std::move(query_keys), query_edges),
    target_(std::move(target_keys), target_edges),
    plan_(build_query_plan(query_)),
    compatibility_(
        query_.node_count(),
        target_.node_count(),
        query_.edge_count(),
        target_.edge_count(),
        node_compatible,
        edge_compatible
    ) {
    const std::size_t word_count = (target_.node_count() + 63) / 64;
    domains_.assign(
        query_.node_count(),
        std::vector<std::uint64_t>(word_count, 0)
    );
    for (Position query_node = 0; query_node < query_.node_count(); ++query_node) {
        for (Position target_node = 0; target_node < target_.node_count(); ++target_node) {
            if (
                query_.degrees()[query_node] <= target_.degrees()[target_node]
                && compatibility_.may_accept_node(query_node, target_node)
            ) {
                domains_[query_node][target_node / 64] |= (
                    std::uint64_t{1} << (target_node % 64)
                );
            }
        }
    }
}

MappingCursor::MappingCursor(std::shared_ptr<const SyntheticProblem> problem)
    : problem_(std::move(problem)),
      query_to_target_(problem_->query().node_count(), -1),
      target_to_query_(problem_->target().node_count(), -1),
      frames_(problem_->query().node_count()) {}

bool MappingCursor::domain_contains(
    const Position query,
    const Position target
) const noexcept {
    return bit_is_set(problem_->domains()[query], target);
}

void MappingCursor::initialize_frame(const std::size_t depth) {
    Frame& frame = frames_[depth];
    const Position query_node = problem_->plan().order[depth];
    const std::int32_t parent = problem_->plan().parent[query_node];
    frame.initialized = true;
    if (parent < 0) {
        frame.cursor = 0;
        frame.end = problem_->target().node_count();
        return;
    }
    const Position parent_target = static_cast<Position>(
        query_to_target_[static_cast<Position>(parent)]
    );
    frame.cursor = problem_->target().offsets()[parent_target];
    frame.end = problem_->target().offsets()[parent_target + 1];
}

std::optional<Position> MappingCursor::next_candidate(const std::size_t depth) {
    Frame& frame = frames_[depth];
    if (!frame.initialized) {
        initialize_frame(depth);
    }
    const Position query_node = problem_->plan().order[depth];
    const bool is_root = problem_->plan().parent[query_node] < 0;
    while (frame.cursor < frame.end) {
        const std::size_t cursor = frame.cursor++;
        const Position target_node = is_root
            ? static_cast<Position>(cursor)
            : problem_->target().neighbors()[cursor];
        if (domain_contains(query_node, target_node)) {
            return target_node;
        }
    }
    return std::nullopt;
}

bool MappingCursor::candidate_matches(
    const Position query_node,
    const Position target_node
) const {
    if (target_to_query_[target_node] >= 0) {
        return false;
    }
    if (!problem_->compatibility().accept_node(query_node, target_node)) {
        return false;
    }

    const std::int32_t query_loop = problem_->query().self_loop_edge(query_node);
    if (query_loop >= 0) {
        const std::int32_t target_loop = problem_->target().self_loop_edge(target_node);
        if (
            target_loop < 0
            || !problem_->compatibility().accept_edge(
                static_cast<EdgeIndex>(query_loop),
                static_cast<EdgeIndex>(target_loop)
            )
        ) {
            return false;
        }
    }

    for (const EarlierEdge& query_edge : problem_->plan().earlier_edges[query_node]) {
        const Position mapped_neighbor = static_cast<Position>(
            query_to_target_[query_edge.neighbor]
        );
        const std::int32_t target_edge = problem_->target().edge_between(
            target_node,
            mapped_neighbor
        );
        if (
            target_edge < 0
            || !problem_->compatibility().accept_edge(
                query_edge.edge,
                static_cast<EdgeIndex>(target_edge)
            )
        ) {
            return false;
        }
    }
    return true;
}

bool MappingCursor::remaining_vertex_has_candidate(const Position query_node) const {
    for (Position target_node = 0; target_node < problem_->target().node_count(); ++target_node) {
        if (
            target_to_query_[target_node] >= 0
            || !domain_contains(query_node, target_node)
        ) {
            continue;
        }
        const std::int32_t query_loop = problem_->query().self_loop_edge(query_node);
        if (query_loop >= 0) {
            const std::int32_t target_loop = problem_->target().self_loop_edge(target_node);
            if (
                target_loop < 0
                || !problem_->compatibility().may_accept_edge(
                    static_cast<EdgeIndex>(query_loop),
                    static_cast<EdgeIndex>(target_loop)
                )
            ) {
                continue;
            }
        }

        bool possible = true;
        for (
            Position adjacency = problem_->query().offsets()[query_node];
            adjacency < problem_->query().offsets()[query_node + 1];
            ++adjacency
        ) {
            const Position query_neighbor = problem_->query().neighbors()[adjacency];
            if (query_neighbor == query_node || query_to_target_[query_neighbor] < 0) {
                continue;
            }
            const Position target_neighbor = static_cast<Position>(
                query_to_target_[query_neighbor]
            );
            const std::int32_t target_edge = problem_->target().edge_between(
                target_node,
                target_neighbor
            );
            if (
                target_edge < 0
                || !problem_->compatibility().may_accept_edge(
                    problem_->query().adjacency_edges()[adjacency],
                    static_cast<EdgeIndex>(target_edge)
                )
            ) {
                possible = false;
                break;
            }
        }
        if (possible) {
            return true;
        }
    }
    return false;
}

bool MappingCursor::forward_checks(
    const Position query_node,
    const Position target_node
) const {
    std::size_t unmapped_query_neighbors = 0;
    for (
        Position adjacency = problem_->query().offsets()[query_node];
        adjacency < problem_->query().offsets()[query_node + 1];
        ++adjacency
    ) {
        const Position neighbor = problem_->query().neighbors()[adjacency];
        if (neighbor != query_node && query_to_target_[neighbor] < 0) {
            ++unmapped_query_neighbors;
        }
    }
    std::size_t unused_target_neighbors = 0;
    for (
        Position adjacency = problem_->target().offsets()[target_node];
        adjacency < problem_->target().offsets()[target_node + 1];
        ++adjacency
    ) {
        const Position neighbor = problem_->target().neighbors()[adjacency];
        if (target_to_query_[neighbor] < 0) {
            ++unused_target_neighbors;
        }
    }
    if (unmapped_query_neighbors > unused_target_neighbors) {
        return false;
    }

    for (Position remaining = 0; remaining < problem_->query().node_count(); ++remaining) {
        if (
            query_to_target_[remaining] < 0
            && !remaining_vertex_has_candidate(remaining)
        ) {
            return false;
        }
    }
    return true;
}

void MappingCursor::assign(
    const Position query,
    const Position target
) noexcept {
    query_to_target_[query] = static_cast<std::int32_t>(target);
    target_to_query_[target] = static_cast<std::int32_t>(query);
}

void MappingCursor::unassign(
    const Position query,
    const Position target
) noexcept {
    query_to_target_[query] = -1;
    target_to_query_[target] = -1;
}

void MappingCursor::backtrack_after_yield() noexcept {
    resume_after_yield_ = false;
    --depth_;
    const Position query_node = problem_->plan().order[depth_];
    const Position target_node = static_cast<Position>(query_to_target_[query_node]);
    unassign(query_node, target_node);
}

std::vector<std::int64_t> MappingCursor::output_mapping() const {
    std::vector<std::int64_t> mapping;
    mapping.reserve(query_to_target_.size());
    for (Position query_node = 0; query_node < query_to_target_.size(); ++query_node) {
        mapping.push_back(problem_->target().keys()[
            static_cast<Position>(query_to_target_[query_node])
        ]);
    }
    return mapping;
}

std::optional<std::vector<std::int64_t>> MappingCursor::next_mapping() {
    AdvancingGuard advancing(advancing_);
    if (exhausted_) {
        return std::nullopt;
    }
    if (problem_->query().node_count() == 0) {
        if (empty_mapping_yielded_) {
            exhausted_ = true;
            return std::nullopt;
        }
        empty_mapping_yielded_ = true;
        return std::vector<std::int64_t>{};
    }
    if (resume_after_yield_) {
        backtrack_after_yield();
    }

    while (true) {
        if (depth_ == problem_->query().node_count()) {
            resume_after_yield_ = true;
            return output_mapping();
        }

        const Position query_node = problem_->plan().order[depth_];
        const std::optional<Position> candidate = next_candidate(depth_);
        if (!candidate.has_value()) {
            frames_[depth_] = Frame{};
            if (depth_ == 0) {
                exhausted_ = true;
                return std::nullopt;
            }
            --depth_;
            const Position previous_query = problem_->plan().order[depth_];
            const Position previous_target = static_cast<Position>(
                query_to_target_[previous_query]
            );
            unassign(previous_query, previous_target);
            continue;
        }

        if (!candidate_matches(query_node, *candidate)) {
            continue;
        }
        assign(query_node, *candidate);
        if (!forward_checks(query_node, *candidate)) {
            unassign(query_node, *candidate);
            continue;
        }
        ++depth_;
        if (depth_ < frames_.size()) {
            frames_[depth_] = Frame{};
        }
    }
}

}  // namespace hotpot::search
