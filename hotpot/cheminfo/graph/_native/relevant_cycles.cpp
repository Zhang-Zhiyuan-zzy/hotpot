#include "relevant_cycles.hpp"

#include <string>


namespace hotpot::graph {

std::vector<CycleEdges> relevant_cycles(
    const std::vector<Edge>& edges,
    const RelevantCycleLimits& limits
) {
    const std::string max_size = limits.max_size
        ? std::to_string(*limits.max_size)
        : "none";
    const std::string max_cycles = limits.max_cycles
        ? std::to_string(*limits.max_cycles)
        : "none";
    throw RelevantCyclesNotImplemented(
        "the Relevant Cycles native interface is connected but the algorithm "
        "has not been implemented; edges=" + std::to_string(edges.size()) +
        ", max_size=" + max_size + ", max_cycles=" + max_cycles
    );
}

}  // namespace hotpot::graph
