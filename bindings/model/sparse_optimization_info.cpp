#include "bindings/model/bindings.h"

// PyBind11 headers
#include <pybind11/iostream.h>

// Timeloop headers
#include "model/sparse-optimization-info.hpp"
#include "model/sparse-optimization-parser.hpp"

namespace model_bindings {

void BindSparseOptimizationInfo(py::module& m) {
  py::class_<sparse::SparseOptimizationInfo>(m, "NativeSparseOptimizationInfo")
      .def(py::init(&sparse::ParseAndConstruct))
      .def_static("parse_and_construct", &sparse::ParseAndConstruct,
                  py::call_guard<py::scoped_ostream_redirect,
                                 py::scoped_estream_redirect>());
}

}  // namespace model_bindings
