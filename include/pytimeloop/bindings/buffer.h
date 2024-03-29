#pragma once

// PyBind11 headers
#include "pybind11/pybind11.h"

#define USE_ACCELERGY

namespace py = pybind11;

namespace pytimeloop::model_bindings {

void BindBufferClasses(py::module& m);

}
