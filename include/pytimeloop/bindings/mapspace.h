#pragma once

// PyBind11 headers
#include "pybind11/iostream.h"
#include "pybind11/pybind11.h"
#include "pybind11/stl.h"

#define USE_ACCELERGY

namespace py = pybind11;

void BindMapspaceClasses(py::module& m);
