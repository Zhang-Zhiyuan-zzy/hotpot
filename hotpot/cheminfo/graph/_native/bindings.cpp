#include "relevant_cycles.hpp"

#include <optional>
#include <utility>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>


namespace py = pybind11;


PYBIND11_MODULE(_relevant_cycles, module) {
    py::register_exception<hotpot::graph::RelevantCycleLimitExceeded>(
        module,
        "RelevantCycleLimitExceeded",
        PyExc_RuntimeError
    );
    py::register_exception<hotpot::graph::RelevantCyclesNotImplemented>(
        module,
        "RelevantCyclesNotImplemented",
        PyExc_NotImplementedError
    );

    module.def(
        "relevant_cycles",
        [](const std::vector<hotpot::graph::Edge>& edges,
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
        "_interface_probe",
        [](const std::vector<hotpot::graph::Edge>& edges,
           const std::optional<std::size_t> max_size,
           const std::optional<std::size_t> max_cycles) {
            return py::make_tuple(edges, max_size, max_cycles);
        },
        py::arg("edges"),
        py::arg("max_size") = std::nullopt,
        py::arg("max_cycles") = std::nullopt
    );
}
