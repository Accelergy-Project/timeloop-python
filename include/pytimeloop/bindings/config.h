#pragma once

// PyBind11 headers
#include "pybind11/pybind11.h"

#define USE_ACCELERGY

namespace py = pybind11;

namespace pytimeloop::config_bindings {

void BindConfigClasses(py::module& m);

}
