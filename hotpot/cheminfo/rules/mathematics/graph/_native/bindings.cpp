#include "cycle_basis.hpp"
#include "graph.hpp"
#include "relevant_cycles.hpp"

#include <optional>
#include <utility>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>


namespace py = pybind11;


PYBIND11_MODULE(_graph_native, module) {
    py::register_exception<hotpot::graph::RelevantCycleLimitExceeded>(
        module,
        "RelevantCycleLimitExceeded",
        PyExc_RuntimeError
    );
    module.def(
        "relevant_cycles",
        [](const std::vector<hotpot::graph::RelevantEdge>& edges,
           const std::optional<std::size_t> max_size,
           const std::optional<std::size_t> max_cycles) {
            return hotpot::graph::relevant_cycles(
                edges,
                hotpot::graph::RelevantCycleLimits{max_size, max_cycles}
            );
        },
        py::arg("edges"),
        py::arg("max_size") = std::nullopt,
        py::arg("max_cycles") = std::nullopt,
        py::call_guard<py::gil_scoped_release>()
    );
    module.def(
        "cycle_basis",
        [](std::vector<hotpot::graph::NodeKey> nodes,
           const std::vector<std::pair<
               hotpot::graph::NodeKey,
               hotpot::graph::NodeKey
           >>& edges) {
            const hotpot::graph::Graph graph(std::move(nodes), edges);
            return hotpot::graph::cycle_basis(graph);
        },
        py::arg("nodes"),
        py::arg("edges"),
        py::call_guard<py::gil_scoped_release>()
    );
    module.def(
        "all_cycle_basis",
        [](std::vector<hotpot::graph::NodeKey> nodes,
           const std::vector<std::pair<
               hotpot::graph::NodeKey,
               hotpot::graph::NodeKey
           >>& edges) {
            const hotpot::graph::Graph graph(std::move(nodes), edges);
            return hotpot::graph::all_cycle_basis(graph);
        },
        py::arg("nodes"),
        py::arg("edges"),
        py::call_guard<py::gil_scoped_release>()
    );
}
