#include "synthetic_compatibility.hpp"

#include <stdexcept>
#include <string>


namespace hotpot::search {

namespace {

void append_matrix(
    const std::vector<std::vector<std::uint8_t>>& rows,
    const std::size_t expected_rows,
    const std::size_t expected_columns,
    const char* name,
    std::vector<std::uint8_t>& destination
) {
    if (rows.size() != expected_rows) {
        throw std::invalid_argument(std::string(name) + " has the wrong row count");
    }
    destination.reserve(expected_rows * expected_columns);
    for (const auto& row : rows) {
        if (row.size() != expected_columns) {
            throw std::invalid_argument(
                std::string(name) + " has the wrong column count"
            );
        }
        for (const std::uint8_t value : row) {
            if (value > 1) {
                throw std::invalid_argument(
                    std::string(name) + " values must be boolean"
                );
            }
            destination.push_back(value);
        }
    }
}

}  // namespace

SyntheticCompatibility::SyntheticCompatibility(
    const std::size_t query_node_count,
    const std::size_t target_node_count,
    const std::size_t query_edge_count,
    const std::size_t target_edge_count,
    const std::vector<std::vector<std::uint8_t>>& node_compatible,
    const std::vector<std::vector<std::uint8_t>>& edge_compatible
) : target_node_count_(target_node_count),
    target_edge_count_(target_edge_count) {
    append_matrix(
        node_compatible,
        query_node_count,
        target_node_count,
        "node_compatible",
        node_compatible_
    );
    append_matrix(
        edge_compatible,
        query_edge_count,
        target_edge_count,
        "edge_compatible",
        edge_compatible_
    );
}

bool SyntheticCompatibility::accept_node(
    const Position query,
    const Position target
) const noexcept {
    return node_compatible_[query * target_node_count_ + target] != 0;
}

bool SyntheticCompatibility::accept_edge(
    const EdgeIndex query,
    const EdgeIndex target
) const noexcept {
    return edge_compatible_[query * target_edge_count_ + target] != 0;
}

}  // namespace hotpot::search
