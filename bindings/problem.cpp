#include "bindings.h"

// Timeloop headers
#include "workload/problem-shape.hpp"
#include "workload/workload.hpp"

void BindProblemClasses(py::module& m) {
  py::class_<problem::Workload>(m, "Workload")
      .def(py::init<>())
      .def("parse_workload",
           [](problem::Workload& w, config::CompoundConfigNode& config) {
             problem::ParseWorkload(config, w);
           });

  py::class_<problem::Shape>(m, "ProblemShape")
      .def_readonly("num_data_spaces", &problem::Shape::NumDataSpaces);

  m.def("get_problem_shape", &problem::GetShape);
}
