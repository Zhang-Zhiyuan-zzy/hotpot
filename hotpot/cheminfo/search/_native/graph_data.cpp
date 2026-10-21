#include "graph_data.hpp"

#include <algorithm>
#include <limits>
#include <stdexcept>
#include <tuple>


namespace hotpot::search {

namespace {

using AdjacencyEntry = std::pair<Position, EdgeIndex>;

}  // namespace

GraphData::GraphData(
    std::vector<std::int64_t> keys,
    const std::vector<std::pair<std::int64_t, std::int64_t>>& input_edges
) : keys_(std::move(keys)) {
    if (keys_.size() > std::numeric_limits<Position>::max()) {
        throw std::invalid_argument("graph contains too many nodes");
    }
    if (!std::is_sorted(keys_.begin(), keys_.end())) {
        throw std::invalid_argument("node keys must be sorted in ascending order");
    }
    if (std::adjacent_find(keys_.begin(), keys_.end()) != keys_.end()) {
        throw std::invalid_argument("node keys must be unique");
    }
    if (input_edges.size() > std::numeric_limits<EdgeIndex>::max()) {
        throw std::invalid_argument("graph contains too many edges");
    }

    std::pair<std::int64_t, std::int64_t> previous_edge{};
    bool has_previous_edge = false;
    edges_.reserve(input_edges.size());
    for (const auto& edge : input_edges) {
        if (edge.first > edge.second) {
            throw std::invalid_argument(
                "edge endpoints must use ascending undirected key order"
            );
        }
        if (has_previous_edge && !(previous_edge < edge)) {
            throw std::invalid_argument(
                "edges must be unique and sorted lexicographically"
            );
        }
        edges_.push_back({
            position_for_key(edge.first),
            position_for_key(edge.second),
        });
        previous_edge = edge;
        has_previous_edge = true;
    }

    std::vector<std::vector<AdjacencyEntry>> adjacency(keys_.size());
    degrees_.assign(keys_.size(), 0);
    self_loop_edges_.assign(keys_.size(), -1);
    for (EdgeIndex edge_index = 0; edge_index < edges_.size(); ++edge_index) {
        const Edge& edge = edges_[edge_index];
        adjacency[edge.first].push_back({edge.second, edge_index});
        if (edge.first == edge.second) {
            degrees_[edge.first] += 2;
            self_loop_edges_[edge.first] = static_cast<std::int32_t>(edge_index);
        } else {
            adjacency[edge.second].push_back({edge.first, edge_index});
            ++degrees_[edge.first];
            ++degrees_[edge.second];
        }
    }

    offsets_.reserve(keys_.size() + 1);
    offsets_.push_back(0);
    for (auto& node_adjacency : adjacency) {
        std::sort(node_adjacency.begin(), node_adjacency.end());
        for (const auto& [neighbor, edge_index] : node_adjacency) {
            neighbors_.push_back(neighbor);
            adjacency_edges_.push_back(edge_index);
        }
        offsets_.push_back(static_cast<Position>(neighbors_.size()));
    }
}

Position GraphData::position_for_key(const std::int64_t key) const {
    const auto found = std::lower_bound(keys_.begin(), keys_.end(), key);
    if (found == keys_.end() || *found != key) {
        throw std::invalid_argument("edge endpoint is not a graph node key");
    }
    return static_cast<Position>(found - keys_.begin());
}

std::int32_t GraphData::self_loop_edge(const Position node) const noexcept {
    return self_loop_edges_[node];
}

std::int32_t GraphData::edge_between(
    const Position first,
    const Position second
) const noexcept {
    const Position begin = offsets_[first];
    const Position end = offsets_[first + 1];
    const auto found = std::lower_bound(
        neighbors_.begin() + begin,
        neighbors_.begin() + end,
        second
    );
    if (found == neighbors_.begin() + end || *found != second) {
        return -1;
    }
    const auto adjacency_index = static_cast<std::size_t>(
        found - neighbors_.begin()
    );
    return static_cast<std::int32_t>(adjacency_edges_[adjacency_index]);
}

}  // namespace hotpot::search
