#pragma once

#include <cstddef>
#include <cstdint>
#include <utility>
#include <vector>


namespace hotpot::graph {

using NodeKey = std::int64_t;
using Position = std::uint32_t;
using EdgeIndex = std::uint32_t;

struct Edge {
    Position first;
    Position second;
};

class Graph final {
public:
    using NeighborIterator = std::vector<Position>::const_iterator;
    using IncidentEdgeIterator = std::vector<EdgeIndex>::const_iterator;

    Graph(
        std::vector<NodeKey> ordered_keys,
        const std::vector<std::pair<NodeKey, NodeKey>>& ordered_edges
    );

    std::size_t node_count() const noexcept { return keys_.size(); }
    std::size_t edge_count() const noexcept { return edges_.size(); }

    const std::vector<NodeKey>& keys() const noexcept { return keys_; }
    const std::vector<Edge>& edges() const noexcept { return edges_; }
    const std::vector<Position>& offsets() const noexcept { return offsets_; }
    const std::vector<Position>& neighbors() const noexcept { return neighbors_; }
    const std::vector<EdgeIndex>& adjacency_edges() const noexcept {
        return adjacency_edges_;
    }
    const std::vector<Position>& degrees() const noexcept { return degrees_; }

    std::pair<NeighborIterator, NeighborIterator> neighbors(
        Position node
    ) const noexcept;
    std::pair<IncidentEdgeIterator, IncidentEdgeIterator> incident_edges(
        Position node
    ) const noexcept;
    Position degree(Position node) const noexcept { return degrees_[node]; }
    std::int32_t self_loop_edge(Position node) const noexcept;
    std::int32_t edge_between(Position first, Position second) const noexcept;

private:
    Position position_for_key(NodeKey key) const;

    std::vector<NodeKey> keys_;
    std::vector<Edge> edges_;
    std::vector<Position> offsets_;
    std::vector<Position> neighbors_;
    std::vector<EdgeIndex> adjacency_edges_;
    std::vector<Position> degrees_;
    std::vector<std::int32_t> self_loop_edges_;
};

}  // namespace hotpot::graph
