#pragma once

#include <cstdint>
#include <optional>
#include <vector>

// Timeloop library
#include <model/level.hpp>

namespace pytimeloop::pymodel {

struct EvaluationResult {
  uint64_t id;
  std::vector<model::EvalStatus> pre_eval_status;
  std::optional<std::vector<model::EvalStatus>> eval_status;
  double utilization;
  double energy;
  double area;
  uint64_t cycles;
  uint64_t algorithmic_computes;
  uint64_t actual_computes;
  uint64_t last_level_accesses;

  static EvaluationResult FailedEvaluation(
      const std::vector<model::EvalStatus>& pre_eval_status, uint64_t id = 0) {
    return EvaluationResult{.id = id,
                            .pre_eval_status = pre_eval_status,
                            .eval_status = std::nullopt,
                            .utilization = 0,
                            .energy = 0,
                            .area = 0,
                            .cycles = 0,
                            .algorithmic_computes = 0,
                            .actual_computes = 0,
                            .last_level_accesses = 0};
  }
};

}  // namespace pytimeloop::pymodel
