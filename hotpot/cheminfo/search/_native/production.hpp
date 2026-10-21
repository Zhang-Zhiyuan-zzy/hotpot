#pragma once

#include "matcher.hpp"

#include <pybind11/pytypes.h>

#include <cstdint>
#include <memory>
#include <vector>


namespace hotpot::search {

namespace py = pybind11;

class NativeTarget final {
public:
    struct Data;

    explicit NativeTarget(const py::dict& target_spec);
    ~NativeTarget();

    NativeTarget(const NativeTarget&) = delete;
    NativeTarget& operator=(const NativeTarget&) = delete;

    const GraphData& graph() const noexcept { return graph_; }
    const Data& data() const noexcept { return *data_; }

private:
    GraphData graph_;
    std::unique_ptr<Data> data_;
};

class NativeProblem final :
    public SearchProblem,
    public std::enable_shared_from_this<NativeProblem> {
public:
    NativeProblem(
        const py::dict& query_spec,
        std::shared_ptr<NativeTarget> target
    );
    ~NativeProblem() override;

    NativeProblem(const NativeProblem&) = delete;
    NativeProblem& operator=(const NativeProblem&) = delete;

    const GraphData& query() const noexcept override { return query_; }
    const GraphData& target() const noexcept override { return target_->graph(); }
    const QueryPlan& plan() const noexcept override { return plan_; }
    const Compatibility& compatibility() const noexcept override {
        return *compatibility_;
    }
    const std::vector<std::vector<std::uint64_t>>& domains() const noexcept override {
        return domains_;
    }

    bool has_anchored_match(Position target_node) const;

private:
    GraphData query_;
    std::shared_ptr<NativeTarget> target_;
    QueryPlan plan_;
    std::unique_ptr<Compatibility> compatibility_;
    std::vector<std::vector<std::uint64_t>> domains_;
};

}  // namespace hotpot::search
