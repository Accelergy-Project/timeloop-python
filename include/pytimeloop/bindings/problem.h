#pragma once

// PyBind11 headers
#include "pybind11/iostream.h"
#include "pybind11/pybind11.h"
#include "pybind11/stl.h"

#define USE_ACCELERGY

namespace py = pybind11;

namespace pytimeloop::problem_bindings {

void BindProblemClasses(py::module& m);

}  // namespace pytimeloop::problem_bindings
