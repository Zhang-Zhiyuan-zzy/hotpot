#include "automorphism.hpp"
#include "matcher.hpp"

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <set>
#include <utility>
#include <vector>

using namespace hotpot::search;
using Key = std::int64_t;
using Pair = std::pair<Key, Key>;
using Mapping = std::vector<Key>;
using Anchor = std::optional<std::pair<Position, Position>>;

namespace {

struct Fixture {
    std::vector<Key> keys;
    std::vector<Pair> edges;
    std::vector<Key> node_labels;
    std::vector<Key> edge_labels;
};

Fixture fixture(const Position count, const std::uint32_t edge_mask, const bool colored) {
    Fixture result;
    for (Position node = 0; node < count; ++node) {
        result.keys.push_back(-17 + 11 * static_cast<Key>(node));
        result.node_labels.push_back(colored ? static_cast<Key>(node % 2) : 0);
    }
    std::uint32_t bit = 0;
    for (Position first = 0; first < count; ++first) {
        for (Position second = first; second < count; ++second, ++bit) {
            if ((edge_mask & (std::uint32_t{1} << bit)) != 0) {
                result.edges.emplace_back(result.keys[first], result.keys[second]);
                result.edge_labels.push_back(colored ? static_cast<Key>((first + second) % 2) : 0);
            }
        }
    }
    return result;
}

class LabelCompatibility final : public Compatibility {
public:
    LabelCompatibility(const Fixture& query, const Fixture& target) : query_(query), target_(target) {}
    bool accept_node(Position query, Position target) const override {
        return query_.node_labels[query] == target_.node_labels[target];
    }
    bool accept_edge(EdgeIndex query, EdgeIndex target) const override {
        return query_.edge_labels[query] == target_.edge_labels[target];
    }
    bool may_accept_node(Position query, Position target) const override {
        return accept_node(query, target);
    }
    bool may_accept_edge(EdgeIndex query, EdgeIndex target) const override {
        return accept_edge(query, target);
    }
private:
    const Fixture& query_;
    const Fixture& target_;
};

class Problem final : public SearchProblem {
public:
    Problem(Fixture query, Fixture target) :
        query_fixture(std::move(query)), target_fixture(std::move(target)),
        query_(query_fixture.keys, query_fixture.edges),
        target_(target_fixture.keys, target_fixture.edges),
        compatibility_(query_fixture, target_fixture), plan_(build_query_plan(query_)),
        domains_(build_candidate_domains(query_, target_, compatibility_)) {}
    const Graph& query() const noexcept override { return query_; }
    const Graph& target() const noexcept override { return target_; }
    const QueryPlan& plan() const noexcept override { return plan_; }
    const Compatibility& compatibility() const noexcept override { return compatibility_; }
    const std::vector<std::vector<std::uint64_t>>& domains() const noexcept override { return domains_; }

    const Fixture query_fixture;
    const Fixture target_fixture;
private:
    Graph query_;
    Graph target_;
    LabelCompatibility compatibility_;
    QueryPlan plan_;
    std::vector<std::vector<std::uint64_t>> domains_;
};

// Enumerate all injections using only fixture keys/edges/labels, independently
// of query plans, candidate domains, Graph adjacency, and RI traversal/pruning.
std::set<Mapping> brute_force(const Fixture& query, const Fixture& target, Anchor anchor) {
    std::set<Mapping> result;
    Mapping mapping(query.keys.size());
    std::vector<bool> used(target.keys.size(), false);
    std::function<void(Position)> visit = [&](Position query_node) {
        if (query_node == query.keys.size()) {
            for (std::size_t edge = 0; edge < query.edges.size(); ++edge) {
                const Pair endpoints = query.edges[edge];
                const auto first = std::find(query.keys.begin(), query.keys.end(), endpoints.first);
                const auto second = std::find(query.keys.begin(), query.keys.end(), endpoints.second);
                const Key mapped_first = mapping[static_cast<std::size_t>(first - query.keys.begin())];
                const Key mapped_second = mapping[static_cast<std::size_t>(second - query.keys.begin())];
                const Pair mapped_edge = std::minmax(mapped_first, mapped_second);
                const auto found = std::find(target.edges.begin(), target.edges.end(), mapped_edge);
                if (found == target.edges.end()) {
                    return;
                }
                const auto target_edge = static_cast<std::size_t>(found - target.edges.begin());
                if (query.edge_labels[edge] != target.edge_labels[target_edge]) {
                    return;
                }
            }
            result.insert(mapping);
            return;
        }
        for (Position target_node = 0; target_node < target.keys.size(); ++target_node) {
            if (used[target_node]
                || (anchor && query_node == anchor->first && target_node != anchor->second)
                || query.node_labels[query_node] != target.node_labels[target_node]) {
                continue;
            }
            used[target_node] = true;
            mapping[query_node] = target.keys[target_node];
            visit(query_node + 1);
            used[target_node] = false;
        }
    };
    visit(0);
    return result;
}

std::vector<Mapping> enumerate(const std::shared_ptr<const SearchProblem>& problem, Anchor anchor) {
    MappingCursor cursor(problem, anchor);
    std::vector<Mapping> result;
    while (const auto mapping = cursor.next_mapping()) {
        result.push_back(*mapping);
    }
    assert(!cursor.next_mapping());
    return result;
}

void check_problem(const Fixture& query, const Fixture& target, Anchor anchor = std::nullopt) {
    const auto problem = std::make_shared<Problem>(query, target);
    const auto expected = brute_force(query, target, anchor);
    const auto actual = enumerate(problem, anchor);
    assert(actual.size() == expected.size());
    assert(std::set<Mapping>(actual.begin(), actual.end()) == expected);
    assert(actual == enumerate(problem, anchor));
}

void exhaustive_small_graphs() {
    for (const bool colored : {false, true}) {
        for (Position query_count = 0; query_count <= 3; ++query_count) {
            const std::uint32_t query_masks = std::uint32_t{1} << (query_count * (query_count + 1) / 2);
            for (Position target_count = 0; target_count <= 3; ++target_count) {
                const std::uint32_t target_masks = std::uint32_t{1} << (target_count * (target_count + 1) / 2);
                for (std::uint32_t query_mask = 0; query_mask < query_masks; ++query_mask) {
                    const Fixture query = fixture(query_count, query_mask, colored);
                    for (std::uint32_t target_mask = 0; target_mask < target_masks; ++target_mask) {
                        check_problem(query, fixture(target_count, target_mask, colored));
                    }
                }
            }
        }
    }
}

void anchors_and_order() {
    const Fixture disconnected = fixture(2, 0, false);
    const Fixture sparse_target = fixture(4, 0, false);
    const auto problem = std::make_shared<Problem>(disconnected, sparse_target);
    const std::vector<Mapping> ordered = {
        {-17, -6}, {-17, 5}, {-17, 16}, {-6, -17}, {-6, 5}, {-6, 16},
        {5, -17}, {5, -6}, {5, 16}, {16, -17}, {16, -6}, {16, 5},
    };
    assert(enumerate(problem, std::nullopt) == ordered);
    for (const bool colored : {false, true}) {
        for (const std::uint32_t query_mask : {0U, 1U, 2U, 5U, 7U}) {
            const auto query = fixture(2, query_mask, colored);
            for (const std::uint32_t target_mask : {0U, 1U, 38U, 82U, 511U, 1023U}) {
                const auto target = fixture(4, target_mask, colored);
                for (Position q = 0; q < 2; ++q) {
                    for (Position t = 0; t < 4; ++t) {
                        check_problem(query, target, std::make_pair(q, t));
                    }
                }
            }
        }
    }
}

void exact_orbits() {
    for (const bool colored : {false, true}) {
        for (std::uint32_t mask = 0; mask < 64; ++mask) {
            const auto input = fixture(3, mask, colored);
            const auto graph = std::make_shared<Graph>(input.keys, input.edges);
            const auto orbit = std::make_shared<NativeAutomorphism>(
                graph, input.node_labels, input.edge_labels
            );
            const auto mappings = brute_force(input, input, std::nullopt);
            for (Position first = 0; first < 3; ++first) {
                for (Position second = 0; second < 3; ++second) {
                    const bool expected = std::any_of(mappings.begin(), mappings.end(), [&](const Mapping& mapping) {
                        return mapping[first] == input.keys[second];
                    });
                    assert(orbit->equivalent(input.keys[first], input.keys[second]) == expected);
                }
            }
        }
    }
    const auto empty = std::make_shared<NativeAutomorphism>(
        std::make_shared<Graph>(std::vector<Key>{}, std::vector<Pair>{}),
        std::vector<Key>{}, std::vector<Key>{}
    );
    assert(enumerate(empty, std::nullopt) == std::vector<Mapping>{Mapping{}});
}

void aliasing_lifetime() {
    struct Owner {
        explicit Owner(bool& destroyed) : graph({-7, 42}, {{-7, 42}}), destroyed_(destroyed) {}
        ~Owner() { destroyed_ = true; }
        Graph graph;
        bool& destroyed_;
    };
    bool destroyed = false;
    auto owner = std::make_shared<Owner>(destroyed);
    const Graph* original = &owner->graph;
    std::weak_ptr<Owner> weak = owner;
    std::shared_ptr<const Graph> alias(owner, original);
    auto orbit = std::make_shared<NativeAutomorphism>(alias, std::vector<Key>{9, 9}, std::vector<Key>{3});
    owner.reset();
    alias.reset();
    assert(!weak.expired());
    assert(!destroyed);
    assert(&orbit->query() == original);
    assert(&orbit->target() == original);
    assert(orbit->equivalent(-7, 42));
    // The cursor must keep both the problem and its aliased graph owner alive.
    {
        MappingCursor cursor(orbit);
        orbit.reset();
        assert(!weak.expired());
        assert(cursor.next_mapping() == std::optional<Mapping>(Mapping{-7, 42}));
        assert(cursor.next_mapping() == std::optional<Mapping>(Mapping{42, -7}));
        assert(!cursor.next_mapping());
        assert(!destroyed);
    }
    assert(weak.expired());
    assert(destroyed);
}

}  // namespace

int main() {
    exhaustive_small_graphs();
    anchors_and_order();
    exact_orbits();
    aliasing_lifetime();
}
