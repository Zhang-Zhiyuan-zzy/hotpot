#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <stdexcept>
#include <vector>


namespace hotpot::graph {

using VertexId = std::uint32_t;
using Edge = std::array<VertexId, 2>;
using CycleEdges = std::vector<std::size_t>;

struct RelevantCycleLimits {
    std::optional<std::size_t> max_size;
    std::optional<std::size_t> max_cycles;
};

class RelevantCycleLimitExceeded : public std::runtime_error {
public:
    using std::runtime_error::runtime_error;
};

class RelevantCyclesNotImplemented : public std::logic_error {
public:
    using std::logic_error::logic_error;
};

std::vector<CycleEdges> relevant_cycles(
    const std::vector<Edge>& edges,
    const RelevantCycleLimits& limits
);

}  // namespace hotpot::graph
