#pragma once

// PyBind11 headers
#include "pybind11/iostream.h"
#include "pybind11/pybind11.h"
#include "pybind11/stl.h"

namespace py = pybind11;

namespace pytimeloop::ir_bindings {
  void BindWorkloadIR(py::module& m);
}  // namespace pytimeloop::ir_bindings