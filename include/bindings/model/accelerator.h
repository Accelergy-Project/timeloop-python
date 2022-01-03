#pragma once

#include <optional>

#include "bindings/model/eval-result.h"

// PyBind11
#include <pybind11/iostream.h>
#include <pybind11/pybind11.h>

// Timeloop headers
#include <mapping/mapping.hpp>
#include <model/engine.hpp>
#include <model/sparse-optimization-info.hpp>
#include <workload/workload.hpp>

class Accelerator {
 public:
  Accelerator(const model::Engine::Specs& arch_specs);

  ~Accelerator();

  EvaluationResult Evaluate(
      Mapping mapping, problem::Workload& workload,
      sparse::SparseOptimizationInfo& sparse_optimizations,
      bool break_on_failure = false);

 private:
  const model::Engine::Specs& arch_specs_;
  model::Engine engine_;
  std::vector<std::string> level_names_;
};
