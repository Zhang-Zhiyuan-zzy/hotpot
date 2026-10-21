#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>


namespace hotpot::search {

class SyntheticCompatibility {
public:
    SyntheticCompatibility(
        std::size_t query_node_count,
        std::size_t target_node_count,
        std::size_t query_edge_count,
        std::size_t target_edge_count,
        const std::vector<std::vector<std::uint8_t>>& node_compatible,
        const std::vector<std::vector<std::uint8_t>>& edge_compatible
    );

    bool accept_node(std::size_t query, std::size_t target) const noexcept;
    bool accept_edge(std::size_t query, std::size_t target) const noexcept;
    bool may_accept_node(std::size_t query, std::size_t target) const noexcept {
        return accept_node(query, target);
    }
    bool may_accept_edge(std::size_t query, std::size_t target) const noexcept {
        return accept_edge(query, target);
    }

private:
    std::size_t target_node_count_;
    std::size_t target_edge_count_;
    std::vector<std::uint8_t> node_compatible_;
    std::vector<std::uint8_t> edge_compatible_;
};

}  // namespace hotpot::search
