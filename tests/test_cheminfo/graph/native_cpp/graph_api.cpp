#include "graph.hpp"

#include <cassert>
#include <cstdint>
#include <stdexcept>
#include <utility>
#include <vector>


using hotpot::graph::Graph;


template <typename Function>
void assert_invalid(Function function) {
    bool raised = false;
    try {
        function();
    } catch (const std::invalid_argument&) {
        raised = true;
    }
    assert(raised);
}


int main() {
    const Graph graph(
        {-4, 7, 100},
        {{-4, 7}, {7, 7}}
    );
    assert(graph.node_count() == 3);
    assert(graph.edge_count() == 2);
    assert(graph.keys() == std::vector<std::int64_t>({-4, 7, 100}));
    assert(graph.degrees() == std::vector<std::uint32_t>({1, 3, 0}));
    assert(graph.offsets() == std::vector<std::uint32_t>({0, 1, 3, 3}));
    assert(graph.neighbors() == std::vector<std::uint32_t>({1, 0, 1}));
    assert(graph.adjacency_edges() == std::vector<std::uint32_t>({0, 0, 1}));
    assert(graph.edge_between(0, 1) == 0);
    assert(graph.edge_between(1, 0) == 0);
    assert(graph.edge_between(1, 1) == 1);
    assert(graph.edge_between(0, 2) == -1);
    assert(graph.self_loop_edge(0) == -1);
    assert(graph.self_loop_edge(1) == 1);
    assert(graph.self_loop_edge(2) == -1);
    assert(graph.degree(1) == 3);
    const auto [neighbors_begin, neighbors_end] = graph.neighbors(1);
    assert(std::vector<std::uint32_t>(neighbors_begin, neighbors_end)
        == std::vector<std::uint32_t>({0, 1}));
    const auto [edges_begin, edges_end] = graph.incident_edges(1);
    assert(std::vector<std::uint32_t>(edges_begin, edges_end)
        == std::vector<std::uint32_t>({0, 1}));

    assert_invalid([] { Graph({1, 0}, {}); });
    assert_invalid([] { Graph({0, 0}, {}); });
    assert_invalid([] { Graph({0}, {{0, 1}}); });
    assert_invalid([] { Graph({0, 1}, {{1, 0}}); });
    assert_invalid([] { Graph({0, 1}, {{0, 1}, {0, 1}}); });
    assert_invalid([] {
        Graph({0, 1, 2}, {{1, 2}, {0, 1}});
    });
}
