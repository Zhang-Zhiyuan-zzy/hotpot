#pragma once

#include <pybind11/pybind11.h>


namespace hotpot::forcefields {


void bind_native_forcefield_contracts(
    pybind11::module_& module,
    PyObject* forcefield_setup_error
);


}  // namespace hotpot::forcefields
