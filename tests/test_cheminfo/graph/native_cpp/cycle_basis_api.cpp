#include "cycle_basis.hpp"
#include "graph.hpp"

#include <cassert>
#include <cstdint>
#include <stdexcept>
#include <utility>
#include <vector>


using hotpot::graph::CycleBasis;
using hotpot::graph::Graph;
using hotpot::graph::NodeCycle;
using hotpot::graph::NodeKey;


int main() {
    const Graph acyclic(
        {0, 1, 2, 3, 4},
        {{0, 1}, {1, 2}, {1, 3}}
    );
    assert(hotpot::graph::cycle_basis(acyclic).empty());
    assert((
        hotpot::graph::all_cycle_basis(acyclic)
        == std::vector<CycleBasis>{CycleBasis{}}
    ));

    const Graph square_diagonal(
        {0, 1, 2, 3},
        {{0, 1}, {0, 2}, {0, 3}, {1, 2}, {2, 3}}
    );
    const CycleBasis expected_default{
        NodeCycle{0, 2, 3},
        NodeCycle{0, 1, 2},
    };
    assert(hotpot::graph::cycle_basis(square_diagonal) == expected_default);
    const std::vector<CycleBasis> square_bases =
        hotpot::graph::all_cycle_basis(square_diagonal);
    assert(square_bases.size() == 3);
    assert(square_bases.front() == expected_default);

    const Graph complete_four(
        {0, 1, 2, 3},
        {{0, 1}, {0, 2}, {0, 3}, {1, 2}, {1, 3}, {2, 3}}
    );
    assert(hotpot::graph::all_cycle_basis(complete_four).size() == 28);
    assert(hotpot::graph::all_cycle_basis(complete_four, 28).size() == 28);

    const Graph disconnected_sparse(
        {-7, 2, 5, 10, 20, 30, 99},
        {{-7, 2}, {-7, 5}, {2, 5}, {10, 20}, {10, 30}, {20, 30}}
    );
    assert(hotpot::graph::cycle_basis(disconnected_sparse) == CycleBasis({
        NodeCycle{10, 20, 30},
        NodeCycle{-7, 2, 5},
    }));

    bool rejected_loop = false;
    try {
        const Graph loop({1}, {{1, 1}});
        static_cast<void>(hotpot::graph::cycle_basis(loop));
    } catch (const std::invalid_argument&) {
        rejected_loop = true;
    }
    assert(rejected_loop);

    bool limited = false;
    try {
        static_cast<void>(hotpot::graph::all_cycle_basis(complete_four, 27));
    } catch (const std::overflow_error&) {
        limited = true;
    }
    assert(limited);
}
