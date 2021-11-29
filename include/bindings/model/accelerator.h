#pragma once

#include <optional>

#include "bindings/model/eval-result.h"

// PyBind11 headers
#include <pybind11/iostream.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

// Timeloop headers
#include <mapping/mapping.hpp>
#include <model/engine.hpp>
#include <model/sparse-optimization-info.hpp>
#include <workload/workload.hpp>

namespace py = pybind11;

namespace model_bindings {
void BindAccelerator(py::module& m);
}  // namespace model_bindings

class Accelerator {
 public:
  Accelerator(const model::Engine::Specs& arch_specs);

  EvaluationResult Evaluate(
      Mapping& mapping, problem::Workload& workload,
      sparse::SparseOptimizationInfo& sparse_optimizations,
      bool break_on_failure = false);

 private:
  const model::Engine::Specs& arch_specs_;
  model::Engine engine_;
  std::vector<std::string> level_names_;
};
