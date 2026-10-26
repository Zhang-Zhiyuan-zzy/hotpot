#include "cycle_basis.hpp"

#include <algorithm>
#include <cstdint>
#include <iterator>
#include <limits>
#include <set>
#include <stdexcept>
#include <utility>
#include <vector>


namespace hotpot::graph {

namespace {

using PositionCycle = std::vector<Position>;
using BitVector = std::vector<std::uint64_t>;

void require_simple_cycle_graph(const Graph& graph) {
    for (const Edge& edge : graph.edges()) {
        if (edge.first == edge.second) {
            throw std::invalid_argument(
                "cycle bases do not support self-loop edges"
            );
        }
    }
}

NodeCycle external_cycle(
    const Graph& graph,
    const PositionCycle& cycle
) {
    NodeCycle external;
    external.reserve(cycle.size());
    for (const Position node : cycle) {
        external.push_back(graph.keys()[node]);
    }
    return external;
}

PositionCycle canonical_cycle(const PositionCycle& cycle) {
    const auto minimum = std::min_element(cycle.begin(), cycle.end());
    const std::size_t offset = static_cast<std::size_t>(minimum - cycle.begin());

    PositionCycle forward;
    PositionCycle backward;
    forward.reserve(cycle.size());
    backward.reserve(cycle.size());
    for (std::size_t step = 0; step < cycle.size(); ++step) {
        forward.push_back(cycle[(offset + step) % cycle.size()]);
        backward.push_back(
            cycle[(offset + cycle.size() - step) % cycle.size()]
        );
    }
    return std::min(forward, backward);
}

std::vector<PositionCycle> simple_cycles(const Graph& graph) {
    std::set<PositionCycle> unique_cycles;
    std::vector<Position> path;
    std::vector<bool> visited(graph.node_count(), false);

    const auto visit = [&](const auto& self,
                           const Position start,
                           const Position current) -> void {
        const auto [begin, end] = graph.neighbors(current);
        for (auto neighbor = begin; neighbor != end; ++neighbor) {
            if (*neighbor == start) {
                if (path.size() >= 3) {
                    unique_cycles.insert(canonical_cycle(path));
                }
            } else if (*neighbor > start && !visited[*neighbor]) {
                visited[*neighbor] = true;
                path.push_back(*neighbor);
                self(self, start, *neighbor);
                path.pop_back();
                visited[*neighbor] = false;
            }
        }
    };

    for (Position start = 0; start < graph.node_count(); ++start) {
        path.assign(1, start);
        visited[start] = true;
        visit(visit, start, start);
        visited[start] = false;
    }

    std::vector<PositionCycle> cycles(
        unique_cycles.begin(),
        unique_cycles.end()
    );
    std::sort(
        cycles.begin(),
        cycles.end(),
        [](const PositionCycle& first, const PositionCycle& second) {
            if (first.size() != second.size()) {
                return first.size() < second.size();
            }
            return first < second;
        }
    );
    return cycles;
}

std::size_t component_count(const Graph& graph) {
    std::vector<bool> visited(graph.node_count(), false);
    std::vector<Position> stack;
    std::size_t count = 0;
    for (Position start = 0; start < graph.node_count(); ++start) {
        if (visited[start]) {
            continue;
        }
        ++count;
        visited[start] = true;
        stack.push_back(start);
        while (!stack.empty()) {
            const Position node = stack.back();
            stack.pop_back();
            const auto [begin, end] = graph.neighbors(node);
            for (auto neighbor = begin; neighbor != end; ++neighbor) {
                if (!visited[*neighbor]) {
                    visited[*neighbor] = true;
                    stack.push_back(*neighbor);
                }
            }
        }
    }
    return count;
}

BitVector edge_mask(
    const Graph& graph,
    const PositionCycle& cycle
) {
    BitVector mask((graph.edge_count() + 63) / 64, 0);
    for (std::size_t index = 0; index < cycle.size(); ++index) {
        const Position first = cycle[index];
        const Position second = cycle[(index + 1) % cycle.size()];
        const std::int32_t edge = graph.edge_between(first, second);
        if (edge < 0) {
            throw std::logic_error("simple cycle contains a missing edge");
        }
        const auto edge_index = static_cast<std::size_t>(edge);
        mask[edge_index / 64] ^= std::uint64_t{1} << (edge_index % 64);
    }
    return mask;
}

int highest_bit(const BitVector& value) {
    for (std::size_t word = value.size(); word > 0; --word) {
        const std::uint64_t bits = value[word - 1];
        if (bits != 0) {
#if defined(__GNUC__) || defined(__clang__)
            return static_cast<int>(
                (word - 1) * 64 + 63 - __builtin_clzll(bits)
            );
#else
            std::uint64_t remaining = bits;
            int position = 0;
            while (remaining >>= 1) {
                ++position;
            }
            return static_cast<int>((word - 1) * 64 + position);
#endif
        }
    }
    return -1;
}

void xor_assign(BitVector& target, const BitVector& source) {
    for (std::size_t word = 0; word < target.size(); ++word) {
        target[word] ^= source[word];
    }
}

bool insert_independent(
    std::vector<BitVector>& pivots,
    BitVector value
) {
    while (true) {
        const int pivot = highest_bit(value);
        if (pivot < 0) {
            return false;
        }
        if (pivots[static_cast<std::size_t>(pivot)].empty()) {
            pivots[static_cast<std::size_t>(pivot)] = std::move(value);
            return true;
        }
        xor_assign(value, pivots[static_cast<std::size_t>(pivot)]);
    }
}

std::vector<PositionCycle> canonical_basis(
    const CycleBasis& basis,
    const Graph& graph
) {
    std::vector<PositionCycle> canonical;
    canonical.reserve(basis.size());
    for (const NodeCycle& cycle : basis) {
        PositionCycle positions;
        positions.reserve(cycle.size());
        for (const NodeKey key : cycle) {
            const auto found = std::lower_bound(
                graph.keys().begin(),
                graph.keys().end(),
                key
            );
            positions.push_back(static_cast<Position>(found - graph.keys().begin()));
        }
        canonical.push_back(canonical_cycle(positions));
    }
    std::sort(
        canonical.begin(),
        canonical.end(),
        [](const PositionCycle& first, const PositionCycle& second) {
            if (first.size() != second.size()) {
                return first.size() < second.size();
            }
            return first < second;
        }
    );
    return canonical;
}

}  // namespace

CycleBasis cycle_basis(const Graph& graph) {
    require_simple_cycle_graph(graph);
    CycleBasis cycles;
    std::vector<bool> remaining(graph.node_count(), true);
    std::size_t remaining_count = graph.node_count();
    const Position missing = std::numeric_limits<Position>::max();
    std::vector<Position> stack;

    while (remaining_count != 0) {
        Position root = static_cast<Position>(graph.node_count() - 1);
        while (!remaining[root]) {
            --root;
        }

        std::vector<Position> predecessor(graph.node_count(), missing);
        std::vector<std::set<Position>> used(graph.node_count());
        std::vector<bool> discovered(graph.node_count(), false);
        predecessor[root] = root;
        discovered[root] = true;
        stack.push_back(root);

        while (!stack.empty()) {
            const Position node = stack.back();
            stack.pop_back();
            const auto [begin, end] = graph.neighbors(node);
            for (auto neighbor = begin; neighbor != end; ++neighbor) {
                if (!discovered[*neighbor]) {
                    predecessor[*neighbor] = node;
                    stack.push_back(*neighbor);
                    discovered[*neighbor] = true;
                    used[*neighbor].insert(node);
                } else if (used[node].count(*neighbor) == 0) {
                    PositionCycle cycle{*neighbor, node};
                    Position ancestor = predecessor[node];
                    while (used[*neighbor].count(ancestor) == 0) {
                        cycle.push_back(ancestor);
                        ancestor = predecessor[ancestor];
                    }
                    cycle.push_back(ancestor);
                    cycles.push_back(external_cycle(graph, cycle));
                    used[*neighbor].insert(node);
                }
            }
        }

        for (Position node = 0; node < graph.node_count(); ++node) {
            if (discovered[node] && remaining[node]) {
                remaining[node] = false;
                --remaining_count;
            }
        }
    }
    return cycles;
}

std::vector<CycleBasis> all_cycle_basis(
    const Graph& graph,
    const std::size_t max_bases
) {
    require_simple_cycle_graph(graph);
    const std::size_t components = component_count(graph);
    const std::size_t rank = graph.edge_count() + components - graph.node_count();
    const CycleBasis default_basis = cycle_basis(graph);
    if (rank == 0) {
        return {default_basis};
    }

    const std::vector<PositionCycle> candidates = simple_cycles(graph);
    std::vector<BitVector> masks;
    masks.reserve(candidates.size());
    for (const PositionCycle& candidate : candidates) {
        masks.push_back(edge_mask(graph, candidate));
    }

    std::vector<std::vector<std::size_t>> basis_positions;
    std::vector<std::size_t> selected;
    const auto enumerate = [&](const auto& self,
                               const std::size_t candidate,
                               const std::vector<BitVector>& pivots) -> void {
        if (selected.size() == rank) {
            basis_positions.push_back(selected);
            if (basis_positions.size() > max_bases) {
                throw std::overflow_error(
                    "all_cycle_basis exceeds the fixed complete-basis limit"
                );
            }
            return;
        }
        if (
            candidate == candidates.size()
            || selected.size() + candidates.size() - candidate < rank
        ) {
            return;
        }

        std::vector<BitVector> included_pivots = pivots;
        if (insert_independent(included_pivots, masks[candidate])) {
            selected.push_back(candidate);
            self(self, candidate + 1, included_pivots);
            selected.pop_back();
        }
        self(self, candidate + 1, pivots);
    };

    enumerate(
        enumerate,
        0,
        std::vector<BitVector>(graph.edge_count())
    );

    const std::vector<PositionCycle> canonical_default = canonical_basis(
        default_basis,
        graph
    );
    std::vector<CycleBasis> remaining_bases;
    remaining_bases.reserve(basis_positions.size());
    bool found_default = false;
    for (const std::vector<std::size_t>& positions : basis_positions) {
        std::vector<PositionCycle> position_basis;
        CycleBasis external_basis;
        position_basis.reserve(rank);
        external_basis.reserve(rank);
        for (const std::size_t position : positions) {
            position_basis.push_back(candidates[position]);
            external_basis.push_back(external_cycle(graph, candidates[position]));
        }
        if (position_basis == canonical_default) {
            found_default = true;
        } else {
            remaining_bases.push_back(std::move(external_basis));
        }
    }
    if (!found_default) {
        throw std::logic_error("Paton basis was not found among simple-cycle bases");
    }
    std::sort(
        remaining_bases.begin(),
        remaining_bases.end(),
        [](const CycleBasis& first, const CycleBasis& second) {
            const auto cycle_less = [](const NodeCycle& left,
                                       const NodeCycle& right) {
                if (left.size() != right.size()) {
                    return left.size() < right.size();
                }
                return left < right;
            };
            return std::lexicographical_compare(
                first.begin(),
                first.end(),
                second.begin(),
                second.end(),
                cycle_less
            );
        }
    );

    std::vector<CycleBasis> bases;
    bases.reserve(remaining_bases.size() + 1);
    bases.push_back(default_basis);
    bases.insert(
        bases.end(),
        std::make_move_iterator(remaining_bases.begin()),
        std::make_move_iterator(remaining_bases.end())
    );
    return bases;
}

}  // namespace hotpot::graph
