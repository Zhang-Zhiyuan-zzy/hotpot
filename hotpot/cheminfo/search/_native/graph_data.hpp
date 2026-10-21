#pragma once

#include <cstddef>
#include <cstdint>
#include <utility>
#include <vector>


namespace hotpot::search {

using Position = std::uint32_t;
using EdgeIndex = std::uint32_t;

struct Edge {
    Position first;
    Position second;
};

class GraphData {
public:
    GraphData(
        std::vector<std::int64_t> keys,
        const std::vector<std::pair<std::int64_t, std::int64_t>>& edges
    );

    std::size_t node_count() const noexcept { return keys_.size(); }
    std::size_t edge_count() const noexcept { return edges_.size(); }

    const std::vector<std::int64_t>& keys() const noexcept { return keys_; }
    const std::vector<Edge>& edges() const noexcept { return edges_; }
    const std::vector<Position>& offsets() const noexcept { return offsets_; }
    const std::vector<Position>& neighbors() const noexcept { return neighbors_; }
    const std::vector<EdgeIndex>& adjacency_edges() const noexcept {
        return adjacency_edges_;
    }
    const std::vector<Position>& degrees() const noexcept { return degrees_; }

    std::int32_t self_loop_edge(Position node) const noexcept;
    std::int32_t edge_between(Position first, Position second) const noexcept;

private:
    Position position_for_key(std::int64_t key) const;

    std::vector<std::int64_t> keys_;
    std::vector<Edge> edges_;
    std::vector<Position> offsets_;
    std::vector<Position> neighbors_;
    std::vector<EdgeIndex> adjacency_edges_;
    std::vector<Position> degrees_;
    std::vector<std::int32_t> self_loop_edges_;
};

}  // namespace hotpot::search
