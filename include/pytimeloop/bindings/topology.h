#pragma once

// PyBind11 headers
#include "pybind11/pybind11.h"

namespace py = pybind11;

namespace pytimeloop::model_bindings {

void BindTopology(py::module& m);

}  // namespace pytimeloop::model_bindings
