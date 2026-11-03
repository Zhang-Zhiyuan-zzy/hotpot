#include "automorphism.hpp"

#include <algorithm>
#include <stdexcept>
#include <utility>

namespace hotpot::search {

namespace {

class ExactLabelCompatibility final : public Compatibility {
public:
    ExactLabelCompatibility(
        const std::vector<std::int64_t>& node_labels,
        const std::vector<std::int64_t>& edge_labels
    ) : node_labels_(node_labels), edge_labels_(edge_labels) {}

    bool accept_node(const Position query, const Position target) const override {
        return node_labels_[query] == node_labels_[target];
    }

    bool accept_edge(const EdgeIndex query, const EdgeIndex target) const override {
        return edge_labels_[query] == edge_labels_[target];
    }

    bool may_accept_node(
        const Position query,
        const Position target
    ) const override {
        return accept_node(query, target);
    }

    bool may_accept_edge(
        const EdgeIndex query,
        const EdgeIndex target
    ) const override {
        return accept_edge(query, target);
    }

private:
    const std::vector<std::int64_t>& node_labels_;
    const std::vector<std::int64_t>& edge_labels_;
};

}  // namespace

NativeAutomorphism::NativeAutomorphism(
    std::shared_ptr<const Graph> target,
    std::vector<std::int64_t> node_labels,
    std::vector<std::int64_t> edge_labels
) : target_(std::move(target)),
    node_labels_(std::move(node_labels)),
    edge_labels_(std::move(edge_labels)),
    plan_(build_query_plan(*target_)),
    compatibility_(std::make_unique<ExactLabelCompatibility>(
        node_labels_,
        edge_labels_
    )) {
    const Graph& graph = *target_;
    if (
        node_labels_.size() != graph.node_count()
        || edge_labels_.size() != graph.edge_count()
    ) {
        throw std::invalid_argument(
            "native automorphism labels must align with target topology"
        );
    }
    const std::size_t word_count = (graph.node_count() + 63) / 64;
    domains_.assign(
        graph.node_count(),
        std::vector<std::uint64_t>(word_count, 0)
    );
    for (Position query_node = 0; query_node < graph.node_count(); ++query_node) {
        for (
            Position target_node = 0;
            target_node < graph.node_count();
            ++target_node
        ) {
            if (
                graph.degrees()[query_node] == graph.degrees()[target_node]
                && node_labels_[query_node] == node_labels_[target_node]
            ) {
                domains_[query_node][target_node / 64] |= (
                    std::uint64_t{1} << (target_node % 64)
                );
            }
        }
    }
}

NativeAutomorphism::~NativeAutomorphism() = default;

bool NativeAutomorphism::equivalent(
    const std::int64_t first,
    const std::int64_t second
) const {
    const std::vector<std::int64_t>& keys = target_->keys();
    const auto first_key = std::lower_bound(keys.begin(), keys.end(), first);
    const auto second_key = std::lower_bound(keys.begin(), keys.end(), second);
    if (
        first_key == keys.end() || *first_key != first
        || second_key == keys.end() || *second_key != second
    ) {
        throw std::invalid_argument("native automorphism root is not a target node");
    }
    const Position first_position = static_cast<Position>(first_key - keys.begin());
    const Position second_position = static_cast<Position>(second_key - keys.begin());
    MappingCursor cursor(
        shared_from_this(),
        std::make_pair(first_position, second_position)
    );
    return cursor.next_mapping().has_value();
}

}  // namespace hotpot::search
