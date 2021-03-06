#include "pytimeloop/bindings/problem.h"

// Timeloop headers
#include "workload/shape-models/problem-shape.hpp"
#include "workload/workload.hpp"

namespace pytimeloop::problem_bindings {

void BindProblemClasses(py::module& m) {
  py::class_<problem::Workload>(m, "Workload")
      .def(py::init<>())
      .def(py::init([](config::CompoundConfigNode& config) {
        auto w = std::make_unique<problem::Workload>();
        problem::ParseWorkload(config, *w);
        return w;
      }))
      .def("parse_workload",
           [](problem::Workload& w, config::CompoundConfigNode& config) {
             problem::ParseWorkload(config, w);
           });

  py::class_<problem::Shape>(m, "ProblemShape")
      .def_readonly("num_data_spaces", &problem::Shape::NumDataSpaces);

  m.def("get_problem_shape", &problem::GetShape);
}

}  // namespace pytimeloop::problem_bindings
