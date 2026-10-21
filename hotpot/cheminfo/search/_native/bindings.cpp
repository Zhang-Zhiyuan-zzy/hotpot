#include "matcher.hpp"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <cstdint>
#include <memory>
#include <utility>
#include <vector>


namespace py = pybind11;
using namespace hotpot::search;


PYBIND11_MODULE(_search_native, module) {
    py::class_<SyntheticProblem, std::shared_ptr<SyntheticProblem>>(
        module,
        "SyntheticProblem"
    )
        .def(
            py::init<
                std::vector<std::int64_t>,
                const std::vector<std::pair<std::int64_t, std::int64_t>>&,
                std::vector<std::int64_t>,
                const std::vector<std::pair<std::int64_t, std::int64_t>>&,
                const std::vector<std::vector<std::uint8_t>>&,
                const std::vector<std::vector<std::uint8_t>>&
            >(),
            py::arg("query_keys"),
            py::arg("query_edges"),
            py::arg("target_keys"),
            py::arg("target_edges"),
            py::arg("node_compatible"),
            py::arg("edge_compatible")
        )
        .def_property_readonly(
            "query_keys",
            [](const SyntheticProblem& problem) {
                return py::tuple(py::cast(problem.query().keys()));
            }
        )
        .def(
            "cursor",
            [](const std::shared_ptr<SyntheticProblem>& problem) {
                return std::make_unique<MappingCursor>(problem);
            }
        );

    py::class_<MappingCursor>(module, "MappingCursor")
        .def(
            "__iter__",
            [](MappingCursor& cursor) -> MappingCursor& { return cursor; },
            py::return_value_policy::reference_internal
        )
        .def(
            "__next__",
            [](MappingCursor& cursor) {
                std::optional<std::vector<std::int64_t>> mapping;
                {
                    py::gil_scoped_release release;
                    mapping = cursor.next_mapping();
                }
                if (!mapping.has_value()) {
                    throw py::stop_iteration();
                }
                return py::tuple(py::cast(*mapping));
            }
        );
}
