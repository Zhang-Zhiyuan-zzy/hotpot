#pragma once

#include "matcher.hpp"

namespace hotpot::search {

class NativeAutomorphism final :
    public SearchProblem,
    public std::enable_shared_from_this<NativeAutomorphism> {
public:
    NativeAutomorphism(
        std::shared_ptr<const Graph> target,
        std::vector<std::int64_t> node_labels,
        std::vector<std::int64_t> edge_labels
    );
    ~NativeAutomorphism() override;

    NativeAutomorphism(const NativeAutomorphism&) = delete;
    NativeAutomorphism& operator=(const NativeAutomorphism&) = delete;

    const Graph& query() const noexcept override { return *target_; }
    const Graph& target() const noexcept override { return *target_; }
    const QueryPlan& plan() const noexcept override { return plan_; }
    const Compatibility& compatibility() const noexcept override {
        return *compatibility_;
    }
    const std::vector<std::vector<std::uint64_t>>& domains() const noexcept override {
        return domains_;
    }

    bool equivalent(std::int64_t first, std::int64_t second) const;

private:
    std::shared_ptr<const Graph> target_;
    std::vector<std::int64_t> node_labels_;
    std::vector<std::int64_t> edge_labels_;
    QueryPlan plan_;
    std::unique_ptr<Compatibility> compatibility_;
    std::vector<std::vector<std::uint64_t>> domains_;
};

}  // namespace hotpot::search
