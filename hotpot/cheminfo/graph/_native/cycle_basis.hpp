#pragma once

#include "graph.hpp"

#include <cstddef>
#include <vector>


namespace hotpot::graph {

using NodeCycle = std::vector<NodeKey>;
using CycleBasis = std::vector<NodeCycle>;

CycleBasis cycle_basis(const Graph& graph);

std::vector<CycleBasis> all_cycle_basis(
    const Graph& graph,
    std::size_t max_bases = 10'000
);

}  // namespace hotpot::graph
