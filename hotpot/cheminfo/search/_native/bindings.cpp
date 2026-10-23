#include "matcher.hpp"
#include "production.hpp"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <cstdint>
#include <memory>
#include <vector>


namespace py = pybind11;
using namespace hotpot::search;


PYBIND11_MODULE(_search_native, module) {
    py::class_<NativeTarget, std::shared_ptr<NativeTarget>>(
        module,
        "NativeTarget"
    )
        .def(
            py::init<const py::dict&>(),
            py::arg("target_spec")
        );

    py::class_<NativeProblem, std::shared_ptr<NativeProblem>>(
        module,
        "NativeProblem"
    )
        .def(
            py::init<const py::dict&, std::shared_ptr<NativeTarget>>(),
            py::arg("query_spec"),
            py::arg("target").none(false)
        )
        .def_property_readonly(
            "query_keys",
            [](const NativeProblem& problem) {
                return py::tuple(py::cast(problem.query().keys()));
            }
        )
        .def(
            "cursor",
            [](const std::shared_ptr<NativeProblem>& problem) {
                return std::make_unique<MappingCursor>(
                    std::static_pointer_cast<const SearchProblem>(problem)
                );
            }
        )
        .def(
            "has_match",
            [](const std::shared_ptr<NativeProblem>& problem) {
                bool matched;
                {
                    py::gil_scoped_release release;
                    MappingCursor cursor(
                        std::static_pointer_cast<const SearchProblem>(problem)
                    );
                    matched = cursor.next_mapping().has_value();
                }
                return matched;
            }
        );

    py::class_<NativeAutomorphism, std::shared_ptr<NativeAutomorphism>>(
        module,
        "_NativeAutomorphism"
    )
        .def(
            py::init<
                std::shared_ptr<NativeTarget>,
                std::vector<std::int64_t>,
                std::vector<std::int64_t>
            >(),
            py::arg("target").none(false),
            py::arg("node_labels"),
            py::arg("edge_labels")
        )
        .def(
            "equivalent",
            &NativeAutomorphism::equivalent,
            py::arg("first"),
            py::arg("second"),
            py::call_guard<py::gil_scoped_release>()
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
