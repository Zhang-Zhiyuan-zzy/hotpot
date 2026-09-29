#pragma once

#include <pybind11/pybind11.h>


namespace hotpot::forcefields {


void bind_native_forcefield_contracts(pybind11::module_& module);


}  // namespace hotpot::forcefields
