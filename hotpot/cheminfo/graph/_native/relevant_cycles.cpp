/*
 * Relevant-cycle perception derived from RingDecomposerLib's implementation
 * of Vismara's algorithm (commit 3a7ff93de0d9c4f6a5661508549c6063573f39c7).
 *
 * RingDecomposerLib is Copyright (c) 2016 University of Hamburg, ZBH,
 * Niek Andresen, Florian Flachsenberg, and Matthias Rarey, and is distributed
 * under the BSD 3-Clause license included with Hotpot's native graph sources.
 */

#include "relevant_cycles.hpp"

#include <algorithm>
#include <cstdint>
#include <functional>
#include <limits>
#include <set>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>


namespace hotpot::graph {
namespace {

using LocalVertex = std::size_t;
using LocalEdge = std::size_t;
using BitWord = std::uint64_t;
using BitVector = std::vector<BitWord>;

constexpr std::size_t BITS_PER_WORD = std::numeric_limits<BitWord>::digits;
constexpr std::size_t NO_INDEX = std::numeric_limits<std::size_t>::max();
constexpr std::size_t INFINITE_DISTANCE = std::numeric_limits<std::size_t>::max();

struct AdjacentEdge {
    LocalVertex vertex;
    LocalEdge edge;
};

struct LocalGraph {
    std::vector<Edge> edges;
    std::vector<std::size_t> original_edge_ids;
    std::vector<std::vector<AdjacentEdge>> adjacency;
};

struct ShortestPathData {
    std::vector<std::vector<std::size_t>> distance;
    std::vector<std::vector<LocalVertex>> predecessor;
    std::vector<std::vector<bool>> ordered_reachable;
    std::vector<std::vector<std::vector<AdjacentEdge>>> path_predecessors;
};

struct CycleFamily {
    LocalVertex root;
    LocalVertex first;
    LocalVertex second;
    LocalVertex center;
    std::size_t weight;
    BitVector prototype;
    bool relevant = false;
};

BitVector make_bit_vector(const std::size_t bit_count) {
    return BitVector((bit_count + BITS_PER_WORD - 1) / BITS_PER_WORD, 0);
}

void set_bit(BitVector& bits, const std::size_t bit) {
    bits[bit / BITS_PER_WORD] |= BitWord{1} << (bit % BITS_PER_WORD);
}

bool test_bit(const BitVector& bits, const std::size_t bit) {
    return (bits[bit / BITS_PER_WORD] >> (bit % BITS_PER_WORD)) & BitWord{1};
}

void xor_bits(BitVector& destination, const BitVector& source) {
    for (std::size_t word = 0; word < destination.size(); ++word) {
        destination[word] ^= source[word];
    }
}

void or_bits(BitVector& destination, const BitVector& source) {
    for (std::size_t word = 0; word < destination.size(); ++word) {
        destination[word] |= source[word];
    }
}

bool bits_empty(const BitVector& bits) {
    return std::all_of(bits.begin(), bits.end(), [](const BitWord word) {
        return word == 0;
    });
}

std::size_t bit_count(const BitVector& bits) {
    std::size_t count = 0;
    for (BitWord word : bits) {
#if defined(__GNUC__) || defined(__clang__)
        count += static_cast<std::size_t>(__builtin_popcountll(word));
#else
        while (word != 0) {
            word &= word - 1;
            ++count;
        }
#endif
    }
    return count;
}

std::size_t highest_set_bit(const BitVector& bits) {
    for (std::size_t word_index = bits.size(); word_index > 0; --word_index) {
        const BitWord word = bits[word_index - 1];
        if (word == 0) {
            continue;
        }
#if defined(__GNUC__) || defined(__clang__)
        const std::size_t offset = BITS_PER_WORD - 1 -
            static_cast<std::size_t>(__builtin_clzll(word));
#else
        std::size_t offset = 0;
        BitWord shifted = word;
        while (shifted >>= 1) {
            ++offset;
        }
#endif
        return (word_index - 1) * BITS_PER_WORD + offset;
    }
    return NO_INDEX;
}

class BinaryCycleBasis {
public:
    explicit BinaryCycleBasis(const std::size_t edge_count)
        : rows_(edge_count) {}

    BitVector reduce(BitVector cycle) const {
        while (!bits_empty(cycle)) {
            const std::size_t pivot = highest_set_bit(cycle);
            if (pivot >= rows_.size() || rows_[pivot].empty()) {
                break;
            }
            xor_bits(cycle, rows_[pivot]);
        }
        return cycle;
    }

    bool insert(BitVector cycle) {
        cycle = reduce(std::move(cycle));
        if (bits_empty(cycle)) {
            return false;
        }
        rows_[highest_set_bit(cycle)] = std::move(cycle);
        return true;
    }

private:
    std::vector<BitVector> rows_;
};

std::size_t inferred_vertex_count(const std::vector<Edge>& edges) {
    VertexId greatest = 0;
    for (const Edge& edge : edges) {
        greatest = std::max(greatest, std::max(edge[0], edge[1]));
    }
    return edges.empty() ? 0 : static_cast<std::size_t>(greatest) + 1;
}

std::vector<std::vector<AdjacentEdge>> make_adjacency(
    const std::size_t vertex_count,
    const std::vector<Edge>& edges
) {
    std::vector<std::vector<AdjacentEdge>> adjacency(vertex_count);
    for (std::size_t edge_id = 0; edge_id < edges.size(); ++edge_id) {
        const LocalVertex first = edges[edge_id][0];
        const LocalVertex second = edges[edge_id][1];
        adjacency[first].push_back({second, edge_id});
        adjacency[second].push_back({first, edge_id});
    }
    for (auto& neighbors : adjacency) {
        std::sort(neighbors.begin(), neighbors.end(), [](const auto& left, const auto& right) {
            return left.vertex < right.vertex;
        });
    }
    return adjacency;
}

void visit_biconnected_components(
    const LocalVertex vertex,
    const LocalEdge parent_edge,
    const std::vector<std::vector<AdjacentEdge>>& adjacency,
    std::vector<std::size_t>& discovery,
    std::vector<std::size_t>& low,
    std::size_t& time,
    std::vector<LocalEdge>& edge_stack,
    std::vector<std::vector<LocalEdge>>& components
) {
    discovery[vertex] = low[vertex] = ++time;
    for (const AdjacentEdge adjacent : adjacency[vertex]) {
        if (adjacent.edge == parent_edge) {
            continue;
        }
        if (discovery[adjacent.vertex] == 0) {
            edge_stack.push_back(adjacent.edge);
            visit_biconnected_components(
                adjacent.vertex,
                adjacent.edge,
                adjacency,
                discovery,
                low,
                time,
                edge_stack,
                components
            );
            low[vertex] = std::min(low[vertex], low[adjacent.vertex]);
            if (low[adjacent.vertex] >= discovery[vertex]) {
                std::vector<LocalEdge> component;
                while (!edge_stack.empty()) {
                    const LocalEdge edge = edge_stack.back();
                    edge_stack.pop_back();
                    component.push_back(edge);
                    if (edge == adjacent.edge) {
                        break;
                    }
                }
                if (component.size() > 1) {
                    components.push_back(std::move(component));
                }
            }
        } else if (discovery[adjacent.vertex] < discovery[vertex]) {
            edge_stack.push_back(adjacent.edge);
            low[vertex] = std::min(low[vertex], discovery[adjacent.vertex]);
        }
    }
}

std::vector<std::vector<LocalEdge>> biconnected_components(
    const std::vector<Edge>& edges
) {
    const std::size_t vertex_count = inferred_vertex_count(edges);
    const auto adjacency = make_adjacency(vertex_count, edges);
    std::vector<std::size_t> discovery(vertex_count, 0);
    std::vector<std::size_t> low(vertex_count, 0);
    std::vector<LocalEdge> edge_stack;
    std::vector<std::vector<LocalEdge>> components;
    std::size_t time = 0;
    for (LocalVertex vertex = 0; vertex < vertex_count; ++vertex) {
        if (discovery[vertex] == 0) {
            visit_biconnected_components(
                vertex,
                NO_INDEX,
                adjacency,
                discovery,
                low,
                time,
                edge_stack,
                components
            );
        }
    }
    return components;
}

LocalGraph make_local_graph(
    const std::vector<Edge>& input_edges,
    std::vector<LocalEdge> component_edge_ids
) {
    std::sort(component_edge_ids.begin(), component_edge_ids.end());
    std::vector<VertexId> vertices;
    vertices.reserve(component_edge_ids.size() * 2);
    for (const LocalEdge edge_id : component_edge_ids) {
        vertices.push_back(input_edges[edge_id][0]);
        vertices.push_back(input_edges[edge_id][1]);
    }
    std::sort(vertices.begin(), vertices.end());
    vertices.erase(std::unique(vertices.begin(), vertices.end()), vertices.end());

    std::unordered_map<VertexId, VertexId> local_vertex;
    local_vertex.reserve(vertices.size());
    for (std::size_t index = 0; index < vertices.size(); ++index) {
        local_vertex.emplace(vertices[index], static_cast<VertexId>(index));
    }

    LocalGraph graph;
    graph.original_edge_ids = std::move(component_edge_ids);
    graph.edges.reserve(graph.original_edge_ids.size());
    for (const LocalEdge edge_id : graph.original_edge_ids) {
        graph.edges.push_back({
            local_vertex.at(input_edges[edge_id][0]),
            local_vertex.at(input_edges[edge_id][1]),
        });
    }
    graph.adjacency = make_adjacency(vertices.size(), graph.edges);
    return graph;
}

bool precedes(
    const LocalGraph& graph,
    const LocalVertex first,
    const LocalVertex second
) {
    const std::size_t first_degree = graph.adjacency[first].size();
    const std::size_t second_degree = graph.adjacency[second].size();
    return first_degree < second_degree ||
        (first_degree == second_degree && first < second);
}

void breadth_first_distances(
    const LocalGraph& graph,
    const LocalVertex root,
    const bool ordered,
    std::vector<std::size_t>& distance,
    std::vector<LocalVertex>& predecessor
) {
    const std::size_t vertex_count = graph.adjacency.size();
    distance.assign(vertex_count, INFINITE_DISTANCE);
    predecessor.assign(vertex_count, NO_INDEX);
    std::vector<LocalVertex> queue(vertex_count);
    std::size_t head = 0;
    std::size_t tail = 0;
    queue[tail++] = root;
    distance[root] = 0;
    predecessor[root] = root;
    while (head < tail) {
        const LocalVertex vertex = queue[head++];
        for (const AdjacentEdge adjacent : graph.adjacency[vertex]) {
            if (ordered && !precedes(graph, adjacent.vertex, root)) {
                continue;
            }
            if (distance[adjacent.vertex] != INFINITE_DISTANCE) {
                continue;
            }
            distance[adjacent.vertex] = distance[vertex] + 1;
            predecessor[adjacent.vertex] = vertex;
            queue[tail++] = adjacent.vertex;
        }
    }
}

ShortestPathData shortest_path_data(const LocalGraph& graph) {
    const std::size_t vertex_count = graph.adjacency.size();
    ShortestPathData paths;
    paths.distance.resize(vertex_count);
    paths.predecessor.resize(vertex_count);
    paths.ordered_reachable.assign(vertex_count, std::vector<bool>(vertex_count, false));
    paths.path_predecessors.resize(
        vertex_count,
        std::vector<std::vector<AdjacentEdge>>(vertex_count)
    );
    for (LocalVertex root = 0; root < vertex_count; ++root) {
        std::vector<std::size_t> ordered_distance;
        std::vector<LocalVertex> ordered_predecessor;
        breadth_first_distances(
            graph,
            root,
            true,
            ordered_distance,
            ordered_predecessor
        );
        for (LocalVertex vertex = 0; vertex < vertex_count; ++vertex) {
            paths.ordered_reachable[root][vertex] =
                ordered_distance[vertex] != INFINITE_DISTANCE;
        }

        std::vector<std::size_t> full_distance;
        std::vector<LocalVertex> full_predecessor;
        breadth_first_distances(
            graph,
            root,
            false,
            full_distance,
            full_predecessor
        );
        paths.distance[root] = full_distance;
        paths.predecessor[root] = ordered_predecessor;
        for (LocalVertex vertex = 0; vertex < vertex_count; ++vertex) {
            if (full_distance[vertex] < ordered_distance[vertex]) {
                paths.predecessor[root][vertex] = full_predecessor[vertex];
            }
        }

        for (LocalVertex vertex = 0; vertex < vertex_count; ++vertex) {
            if (!paths.ordered_reachable[root][vertex] || vertex == root) {
                continue;
            }
            for (const AdjacentEdge adjacent : graph.adjacency[vertex]) {
                if (paths.ordered_reachable[root][adjacent.vertex] &&
                    full_distance[adjacent.vertex] + 1 == full_distance[vertex]) {
                    paths.path_predecessors[root][vertex].push_back(adjacent);
                }
            }
        }
    }
    return paths;
}

bool canonical_paths_share_only_root(
    const LocalVertex root,
    const LocalVertex first,
    const LocalVertex second,
    const ShortestPathData& paths
) {
    std::vector<bool> first_path(paths.predecessor.size(), false);
    LocalVertex vertex = first;
    while (true) {
        first_path[vertex] = true;
        if (vertex == root) {
            break;
        }
        vertex = paths.predecessor[root][vertex];
    }
    vertex = second;
    while (vertex != root) {
        if (first_path[vertex]) {
            return false;
        }
        vertex = paths.predecessor[root][vertex];
    }
    return true;
}

LocalEdge edge_between(
    const LocalGraph& graph,
    const LocalVertex first,
    const LocalVertex second
) {
    const auto adjacent = std::find_if(
        graph.adjacency[first].begin(),
        graph.adjacency[first].end(),
        [second](const AdjacentEdge edge) { return edge.vertex == second; }
    );
    if (adjacent == graph.adjacency[first].end()) {
        throw std::logic_error("cycle construction requested a missing edge");
    }
    return adjacent->edge;
}

void add_canonical_path(
    BitVector& prototype,
    const LocalGraph& graph,
    const ShortestPathData& paths,
    const LocalVertex root,
    LocalVertex vertex
) {
    while (vertex != root) {
        const LocalVertex parent = paths.predecessor[root][vertex];
        set_bit(prototype, edge_between(graph, vertex, parent));
        vertex = parent;
    }
}

CycleFamily make_family(
    const LocalGraph& graph,
    const ShortestPathData& paths,
    const LocalVertex root,
    const LocalVertex first,
    const LocalVertex second,
    const LocalVertex center,
    const LocalEdge first_connector,
    const LocalEdge second_connector
) {
    BitVector prototype = make_bit_vector(graph.edges.size());
    add_canonical_path(prototype, graph, paths, root, first);
    add_canonical_path(prototype, graph, paths, root, second);
    set_bit(prototype, first_connector);
    if (second_connector != NO_INDEX) {
        set_bit(prototype, second_connector);
    }
    return {
        root,
        first,
        second,
        center,
        bit_count(prototype),
        std::move(prototype),
        false,
    };
}

std::vector<CycleFamily> cycle_families(
    const LocalGraph& graph,
    const ShortestPathData& paths
) {
    std::vector<CycleFamily> families;
    const std::size_t vertex_count = graph.adjacency.size();
    for (LocalVertex root = 0; root < vertex_count; ++root) {
        for (LocalVertex vertex = 0; vertex < vertex_count; ++vertex) {
            if (!paths.ordered_reachable[root][vertex]) {
                continue;
            }
            std::vector<AdjacentEdge> predecessors;
            for (const AdjacentEdge adjacent : graph.adjacency[vertex]) {
                if (!paths.ordered_reachable[root][adjacent.vertex]) {
                    continue;
                }
                if (paths.distance[root][adjacent.vertex] + 1 ==
                    paths.distance[root][vertex]) {
                    predecessors.push_back(adjacent);
                    continue;
                }
                if (paths.distance[root][adjacent.vertex] ==
                    paths.distance[root][vertex] + 1) {
                    continue;
                }
                if (precedes(graph, adjacent.vertex, vertex) &&
                    canonical_paths_share_only_root(
                        root,
                        vertex,
                        adjacent.vertex,
                        paths
                    )) {
                    families.push_back(make_family(
                        graph,
                        paths,
                        root,
                        vertex,
                        adjacent.vertex,
                        NO_INDEX,
                        adjacent.edge,
                        NO_INDEX
                    ));
                }
            }
            for (std::size_t first_index = 0;
                 first_index < predecessors.size();
                 ++first_index) {
                for (std::size_t second_index = first_index + 1;
                     second_index < predecessors.size();
                     ++second_index) {
                    const AdjacentEdge first = predecessors[first_index];
                    const AdjacentEdge second = predecessors[second_index];
                    if (canonical_paths_share_only_root(
                        root,
                        first.vertex,
                        second.vertex,
                        paths
                    )) {
                        families.push_back(make_family(
                            graph,
                            paths,
                            root,
                            first.vertex,
                            second.vertex,
                            vertex,
                            first.edge,
                            second.edge
                        ));
                    }
                }
            }
        }
    }
    std::stable_sort(families.begin(), families.end(), [](const auto& left, const auto& right) {
        return left.weight < right.weight;
    });
    return families;
}

void mark_relevant_families(
    std::vector<CycleFamily>& families,
    const std::size_t edge_count
) {
    BinaryCycleBasis basis(edge_count);
    std::size_t group_start = 0;
    while (group_start < families.size()) {
        std::size_t group_end = group_start + 1;
        while (group_end < families.size() &&
               families[group_end].weight == families[group_start].weight) {
            ++group_end;
        }
        for (std::size_t index = group_start; index < group_end; ++index) {
            families[index].relevant =
                !bits_empty(basis.reduce(families[index].prototype));
        }
        for (std::size_t index = group_start; index < group_end; ++index) {
            if (families[index].relevant) {
                basis.insert(families[index].prototype);
            }
        }
        group_start = group_end;
    }
}

void enumerate_paths(
    const LocalVertex root,
    const LocalVertex vertex,
    const ShortestPathData& paths,
    BitVector& path,
    const std::function<void(const BitVector&)>& visit
) {
    if (vertex == root) {
        visit(path);
        return;
    }
    for (const AdjacentEdge predecessor : paths.path_predecessors[root][vertex]) {
        set_bit(path, predecessor.edge);
        enumerate_paths(root, predecessor.vertex, paths, path, visit);
        path[predecessor.edge / BITS_PER_WORD] ^=
            BitWord{1} << (predecessor.edge % BITS_PER_WORD);
    }
}

CycleEdges original_edge_ids(
    const BitVector& cycle,
    const LocalGraph& graph
) {
    CycleEdges result;
    result.reserve(bit_count(cycle));
    for (LocalEdge edge = 0; edge < graph.edges.size(); ++edge) {
        if (test_bit(cycle, edge)) {
            result.push_back(graph.original_edge_ids[edge]);
        }
    }
    std::sort(result.begin(), result.end());
    return result;
}

void add_expanded_family(
    const CycleFamily& family,
    const LocalGraph& graph,
    const ShortestPathData& paths,
    const RelevantCycleLimits& limits,
    std::set<CycleEdges>& cycles
) {
    if (limits.max_size && family.weight > *limits.max_size) {
        return;
    }
    std::vector<BitVector> second_paths;
    BitVector current_second = make_bit_vector(graph.edges.size());
    enumerate_paths(
        family.root,
        family.second,
        paths,
        current_second,
        [&second_paths](const BitVector& path) { second_paths.push_back(path); }
    );

    BitVector current_first = make_bit_vector(graph.edges.size());
    enumerate_paths(
        family.root,
        family.first,
        paths,
        current_first,
        [&](const BitVector& first_path) {
            for (const BitVector& second_path : second_paths) {
                BitVector cycle = first_path;
                or_bits(cycle, second_path);
                if (family.center == NO_INDEX) {
                    set_bit(cycle, edge_between(graph, family.first, family.second));
                } else {
                    set_bit(cycle, edge_between(graph, family.first, family.center));
                    set_bit(cycle, edge_between(graph, family.second, family.center));
                }
                if (bit_count(cycle) != family.weight) {
                    throw std::logic_error(
                        "Vismara family expansion produced a non-simple cycle"
                    );
                }
                cycles.insert(original_edge_ids(cycle, graph));
                if (limits.max_cycles && cycles.size() > *limits.max_cycles) {
                    throw RelevantCycleLimitExceeded(
                        "Relevant Cycle count exceeds max_cycles=" +
                        std::to_string(*limits.max_cycles)
                    );
                }
            }
        }
    );
}

}  // namespace

std::vector<CycleEdges> relevant_cycles(
    const std::vector<Edge>& edges,
    const RelevantCycleLimits& limits
) {
    if (edges.empty()) {
        return {};
    }
    std::set<CycleEdges> unique_cycles;
    for (auto component_edge_ids : biconnected_components(edges)) {
        const LocalGraph graph = make_local_graph(edges, std::move(component_edge_ids));
        const ShortestPathData paths = shortest_path_data(graph);
        std::vector<CycleFamily> families = cycle_families(graph, paths);
        mark_relevant_families(families, graph.edges.size());
        for (const CycleFamily& family : families) {
            if (family.relevant) {
                add_expanded_family(family, graph, paths, limits, unique_cycles);
            }
        }
    }
    std::vector<CycleEdges> result(unique_cycles.begin(), unique_cycles.end());
    std::sort(result.begin(), result.end(), [](const auto& left, const auto& right) {
        if (left.size() != right.size()) {
            return left.size() < right.size();
        }
        return left < right;
    });
    return result;
}

}  // namespace hotpot::graph
