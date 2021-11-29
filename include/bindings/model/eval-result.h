#pragma once

#include <cstdint>
#include <optional>
#include <vector>

// Timeloop headers
#include <model/level.hpp>

// PyBind11 headers
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;
namespace model_bindings {
void BindEvaluationResult(py::module& m);
}  // namespace model_bindings

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
};
