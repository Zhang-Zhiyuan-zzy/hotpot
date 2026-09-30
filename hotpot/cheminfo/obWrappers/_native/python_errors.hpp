#pragma once

#include "native_engine.hpp"

#include <pybind11/pybind11.h>


namespace hotpot::obwrappers {


[[noreturn]] inline void raise_forcefield_setup_error(
    PyObject* exception_type,
    const ForceFieldSetupFailure& error,
    const char* workflow_stage,
    pybind11::handle completed_coordination = pybind11::none()
) {
    namespace py = pybind11;
    py::object instance = py::reinterpret_borrow<py::object>(exception_type)(
        error.what()
    );
    instance.attr("forcefield") = error.forcefield();
    instance.attr("stage") = error.stage();
    instance.attr("backend_stage") = error.stage();
    instance.attr("workflow_stage") = workflow_stage;
    instance.attr("completed_coordination") = completed_coordination;
    PyErr_SetObject(exception_type, instance.ptr());
    throw py::error_already_set();
}


}  // namespace hotpot::obwrappers
