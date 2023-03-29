#include "pytimeloop/bindings/ir.h"

#include "loop-analysis/isl-ir.hpp"

namespace pytimeloop::ir_bindings {
  using namespace analysis;

  void BindWorkloadIR(py::module& m) {
    py::class_<WorkloadIR>(m, "WorkloadIR")
      .def(py::init<>())
      .def("new_einsum", &WorkloadIR::NewEinsum)
      .def("new_data_space", &WorkloadIR::NewDataSpace)
      .def("add_read_dependency", &WorkloadIR::AddReadDependency)
      .def("add_write_dependency", &WorkloadIR::AddWriteDependency)
      .def("add_operation_space_bounds", &WorkloadIR::AddOperationSpaceBounds)
      .def("add_data_space_bounds", &WorkloadIR::AddDataSpaceBounds);
  }
}