#pragma once

#include "compatibility.hpp"

#include <cstddef>
#include <cstdint>
#include <vector>


namespace hotpot::search {

class SyntheticCompatibility final : public Compatibility {
public:
    SyntheticCompatibility(
        std::size_t query_node_count,
        std::size_t target_node_count,
        std::size_t query_edge_count,
        std::size_t target_edge_count,
        const std::vector<std::vector<std::uint8_t>>& node_compatible,
        const std::vector<std::vector<std::uint8_t>>& edge_compatible
    );

    bool accept_node(Position query, Position target) const noexcept override;
    bool accept_edge(EdgeIndex query, EdgeIndex target) const noexcept override;
    bool may_accept_node(Position query, Position target) const noexcept override {
        return accept_node(query, target);
    }
    bool may_accept_edge(EdgeIndex query, EdgeIndex target) const noexcept override {
        return accept_edge(query, target);
    }

private:
    std::size_t target_node_count_;
    std::size_t target_edge_count_;
    std::vector<std::uint8_t> node_compatible_;
    std::vector<std::uint8_t> edge_compatible_;
};

}  // namespace hotpot::search
