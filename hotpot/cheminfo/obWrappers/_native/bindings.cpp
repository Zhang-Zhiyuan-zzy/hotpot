#include "registry.hpp"

#include <optional>
#include <vector>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>


namespace py = pybind11;
using namespace hotpot::obwrappers;


PYBIND11_MODULE(_ob_rules, module) {
    py::enum_<RuleStage>(module, "RuleStage")
        .value("PRE_BUILD", RuleStage::PRE_BUILD)
        .value(
            "PRE_FORCEFIELD_SETUP",
            RuleStage::PRE_FORCEFIELD_SETUP
        )
        .export_values();

    py::class_<AtomSnapshot>(module, "AtomSnapshot")
        .def(
            py::init<int, int, int, bool>(),
            py::arg("atomic_number"),
            py::arg("formal_charge"),
            py::arg("hybridization"),
            py::arg("is_metal")
        )
        .def_readonly("atomic_number", &AtomSnapshot::atomic_number)
        .def_readonly("formal_charge", &AtomSnapshot::formal_charge)
        .def_readonly("hybridization", &AtomSnapshot::hybridization)
        .def_readonly("is_metal", &AtomSnapshot::is_metal);

    py::class_<BondSnapshot>(module, "BondSnapshot")
        .def(
            py::init<std::size_t, std::size_t, int, bool>(),
            py::arg("begin"),
            py::arg("end"),
            py::arg("order"),
            py::arg("aromatic")
        )
        .def_readonly("begin", &BondSnapshot::begin)
        .def_readonly("end", &BondSnapshot::end)
        .def_readonly("order", &BondSnapshot::order)
        .def_readonly("aromatic", &BondSnapshot::aromatic);

    py::class_<RuleDescriptor>(module, "RuleDescriptor")
        .def_readonly("rule_id", &RuleDescriptor::rule_id)
        .def_readonly("version", &RuleDescriptor::version)
        .def_readonly("stage", &RuleDescriptor::stage)
        .def_readonly("priority", &RuleDescriptor::priority);

    py::class_<HybridizationChange>(module, "HybridizationChange")
        .def_readonly("atom_index", &HybridizationChange::atom_index)
        .def_readonly("before", &HybridizationChange::before)
        .def_readonly("after", &HybridizationChange::after);

    py::class_<CoordinateChange>(module, "CoordinateChange")
        .def_readonly("atom_index", &CoordinateChange::atom_index)
        .def_readonly("before", &CoordinateChange::before)
        .def_readonly("after", &CoordinateChange::after);

    py::class_<RuleApplication>(module, "RuleApplication")
        .def_readonly("rule_id", &RuleApplication::rule_id)
        .def_readonly("version", &RuleApplication::version)
        .def_readonly("stage", &RuleApplication::stage)
        .def_readonly("priority", &RuleApplication::priority)
        .def_readonly("atom_indices", &RuleApplication::atom_indices)
        .def_readonly("metric_before", &RuleApplication::metric_before)
        .def_readonly(
            "hybridization_changes",
            &RuleApplication::hybridization_changes
        )
        .def_readonly(
            "coordinate_changes",
            &RuleApplication::coordinate_changes
        );

    py::class_<RulePlan>(module, "RulePlan")
        .def_readonly("stage", &RulePlan::stage)
        .def_readonly("applications", &RulePlan::applications);

    py::register_exception<RuleApplicationLimitExceeded>(
        module,
        "RuleApplicationLimitExceeded",
        PyExc_RuntimeError
    );

    module.def(
        "available_rules",
        [](std::optional<RuleStage> stage) {
            return rule_registry().descriptors(stage);
        },
        py::arg("stage") = std::nullopt,
        py::call_guard<py::gil_scoped_release>()
    );
    module.def(
        "plan_build",
        &plan_build,
        py::arg("atoms"),
        py::arg("bonds"),
        py::call_guard<py::gil_scoped_release>()
    );
    module.def(
        "plan_optimization",
        &plan_optimization,
        py::arg("atoms"),
        py::arg("bonds"),
        py::arg("coordinates"),
        py::arg("singularity_threshold"),
        py::arg("repair_angle_radians"),
        py::call_guard<py::gil_scoped_release>()
    );
}
