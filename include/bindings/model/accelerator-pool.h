#pragma once

#include <condition_variable>
#include <optional>
#include <thread>
#include <vector>

// Timeloop headers
#include <model/engine.hpp>

// Pybind11 headers
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "bindings/model/eval-result.h"

namespace py = pybind11;

namespace model_bindings {
void BindAcceleratorPool(py::module &m);
}  // namespace model_bindings

class AcceleratorPool {
 public:
  AcceleratorPool(model::Engine::Specs arch_specs, unsigned num_workers);

  ~AcceleratorPool();

  uint64_t Evaluate(Mapping mapping, const problem::Workload workload,
                    const sparse::SparseOptimizationInfo sparse_optimizations,
                    bool break_on_failure = false,
                    bool auto_bypass_on_failure = true);

  EvaluationResult GetResult();

 private:
  struct EvaluationTask {
    uint64_t id;
    Mapping mapping;
    problem::Workload workload;
    sparse::SparseOptimizationInfo sparse_optimizations;
    bool break_on_failure;
    bool auto_bypass_on_failure;
  };

  model::Engine::Specs arch_specs_;

  std::vector<std::thread> workers_;
  std::condition_variable worker_cv_;
  std::atomic<bool> terminate_;

  uint64_t cur_id_;

  std::mutex pool_mutex_;  // protects task_q_ as well as terminate_
  std::queue<EvaluationTask> task_q_;

  std::condition_variable result_cv_;
  std::mutex result_mutex_;
  std::queue<EvaluationResult> result_q_;

  void worker_loop();

  void queue_result(EvaluationResult eval_result);

  void stop_workers();
};
