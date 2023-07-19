#pragma once

#include "pybind11/iostream.h"
#include "pybind11/pybind11.h"
#include "pybind11/stl.h"       // Allows autocasting for some std objects.

namespace py = pybind11;

namespace pytimeloop::model_bindings {

void BindAccelerator(py::module& m);
void BindAcceleratorPool(py::module& m);
void BindEngine(py::module& m);
void BindEvaluationResult(py::module& m);
void BindLevel(py::module& m);
void BindSparseOptimizationInfo(py::module& m);
void BindSparseOptimizationParser(py::module& m);

}  // namespace pytimeloop::model_bindings
