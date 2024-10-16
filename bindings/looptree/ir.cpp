#include "pytimeloop/bindings/looptree.h"

#include <sstream>

#include <applications/looptree-model/model.hpp>
#include <workload/fused-workload.hpp>
#include <workload/fused-workload-dependency-analyzer.hpp>

#include <pybind11/stl.h>


#define FUSED_WORKLOAD_METHOD(python_name, cpp_name) \
    def(#python_name, &problem::FusedWorkload::cpp_name)

#define FUSED_WORKLOAD_ANALYZER_METHOD(python_name, cpp_name) \
    def(#python_name, &problem::FusedWorkloadDependencyAnalyzer::cpp_name)

namespace py = pybind11;

#define DEFINE_REPR_VIA_STRINGSTREAM(class) \
  def("__repr__", &print_via_stringstream<class>)

#define DEFINE_PROPERTY(class, name) \
  def_readwrite(#name, &analysis::class::name)


template<typename T>
std::string print_via_stringstream(const T& t)
{
  std::stringstream buf;
  buf << t;
  return buf.str();
}


namespace pytimeloop::looptree_bindings
{

void BindIr(py::module& m)
{
  py::class_<analysis::Temporal>(m, "Temporal")
    .def(py::init<>())
    .DEFINE_REPR_VIA_STRINGSTREAM(analysis::Temporal);

  py::class_<analysis::Spatial>(m, "Spatial")
    .def(py::init<int, analysis::BufferId>())
    .DEFINE_REPR_VIA_STRINGSTREAM(analysis::Spatial);

  py::class_<analysis::Sequential>(m, "Sequential")
    .def(py::init<>())
    .DEFINE_REPR_VIA_STRINGSTREAM(analysis::Sequential);

  py::class_<analysis::PipelineTemporal>(m, "PipelineTemporal")
    .def(py::init<>())
    .DEFINE_REPR_VIA_STRINGSTREAM(analysis::PipelineTemporal);

  py::class_<analysis::PipelineSpatial>(m, "PipelineSpatial")
    .def(py::init<>())
    .DEFINE_REPR_VIA_STRINGSTREAM(analysis::PipelineSpatial);

  py::class_<analysis::LogicalBuffer>(m, "LogicalBuffer")
    .def(py::init<>())
    .DEFINE_PROPERTY(LogicalBuffer, buffer_id)
    .DEFINE_PROPERTY(LogicalBuffer, dspace_id)
    .DEFINE_PROPERTY(LogicalBuffer, branch_leaf_id)
    .DEFINE_REPR_VIA_STRINGSTREAM(analysis::LogicalBuffer);

  py::class_<analysis::Occupancy>(m, "Occupancy")
    .def(py::init<>())
    .DEFINE_PROPERTY(Occupancy, dim_in_tags);

  py::class_<analysis::Fill>(m, "Fill")
    .def(py::init<>())
    .DEFINE_PROPERTY(Fill, dim_in_tags);
}

}