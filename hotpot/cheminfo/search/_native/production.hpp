#pragma once

#include "matcher.hpp"

#include <pybind11/pytypes.h>

#include <cstdint>
#include <memory>
#include <vector>


namespace hotpot::search {

namespace py = pybind11;

class NativeProblem final :
    public SearchProblem,
    public std::enable_shared_from_this<NativeProblem> {
public:
    NativeProblem(const py::dict& query_spec, const py::dict& target_spec);
    ~NativeProblem() override;

    NativeProblem(const NativeProblem&) = delete;
    NativeProblem& operator=(const NativeProblem&) = delete;

    const GraphData& query() const noexcept override { return query_; }
    const GraphData& target() const noexcept override { return target_; }
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
    GraphData target_;
    QueryPlan plan_;
    std::unique_ptr<Compatibility> compatibility_;
    std::vector<std::vector<std::uint64_t>> domains_;
};

}  // namespace hotpot::search
