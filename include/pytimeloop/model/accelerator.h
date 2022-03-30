#pragma once

#include <optional>

#include "pytimeloop/model/eval-result.h"

// Timeloop headers
#include <mapping/mapping.hpp>
#include <model/engine.hpp>
#include <model/sparse-optimization-info.hpp>
#include <workload/workload.hpp>

namespace pytimeloop::pymodel {

class Accelerator {
 public:
  Accelerator(const model::Engine::Specs& arch_specs);

  EvaluationResult Evaluate(
      Mapping mapping, problem::Workload& workload,
      sparse::SparseOptimizationInfo& sparse_optimizations,
      bool break_on_failure = false);

 private:
  const model::Engine::Specs& arch_specs_;
  model::Engine engine_;
  std::vector<std::string> level_names_;
};

}  // namespace pytimeloop::pymodel
