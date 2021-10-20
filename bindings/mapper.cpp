#include <atomic>
#include <condition_variable>
#include <mutex>
#include <optional>
#include <shared_mutex>
#include <thread>

#include "bindings/bindings.h"

// Timeloop headers
#include "mapping/mapping.hpp"
#include "model/engine.hpp"
#include "model/sparse-optimization-info.hpp"
#include "workload/workload.hpp"

namespace {
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

struct EvaluationTask {
  uint64_t id;
  Mapping mapping;
  problem::Workload workload;
  sparse::SparseOptimizationInfo sparse_optimizations;
  bool break_on_failure;
  bool auto_bypass_on_failure;
};

class AcceleratorPool {
 public:
  AcceleratorPool(model::Engine::Specs arch_specs, unsigned num_workers)
      : arch_specs_(arch_specs), terminate_(false), cur_id_(0) {
    for (int i = 0; i < num_workers; i++) {
      workers_.push_back(std::thread(&AcceleratorPool::worker_loop, this));
    }
  };

  ~AcceleratorPool() { stop_workers(); }

  uint64_t Evaluate(Mapping mapping, const problem::Workload workload,
                    const sparse::SparseOptimizationInfo sparse_optimizations,
                    bool break_on_failure = false,
                    bool auto_bypass_on_failure = true) {
    uint64_t task_id;
    {
      std::lock_guard<std::mutex> lock(pool_mutex_);
      task_id = cur_id_++;
      task_q_.push(EvaluationTask{task_id, mapping, workload,
                                  sparse_optimizations, break_on_failure,
                                  auto_bypass_on_failure});
    }
    worker_cv_.notify_one();
    return task_id;
  }

  EvaluationResult GetResult() {
    std::unique_lock<std::mutex> lock(result_mutex_);
    result_cv_.wait(lock,
                    [this]() { return !result_q_.empty() || terminate_; });

    EvaluationResult res = result_q_.front();
    result_q_.pop();
    return res;
  }

  std::optional<EvaluationResult> TryGetResult(float timeout) {
    std::chrono::duration<float> timeout_s{timeout};
    std::unique_lock<std::mutex> lock(result_mutex_);
    if (result_cv_.wait_for(lock, timeout_s, [this]() {
          return !result_q_.empty() || terminate_;
        })) {
      EvaluationResult res = result_q_.front();
      result_q_.pop();
      return res;
    } else {
      return std::nullopt;
    }
  }

 private:
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

  void worker_loop() {
    model::Engine engine;
    engine.Spec(arch_specs_);

    auto level_names = arch_specs_.topology.LevelNames();

    while (true) {
      // Acquire task from task_q_
      std::unique_lock<std::mutex> lock(pool_mutex_);
      worker_cv_.wait(lock,
                      [this]() { return !task_q_.empty() || terminate_; });

      if (terminate_) {
        lock.unlock();
        break;
      }

      auto task = task_q_.front();
      task_q_.pop();
      lock.unlock();

      // TODO: Perform evaluation
      std::vector<model::EvalStatus> pre_eval_status;
      pre_eval_status = engine.PreEvaluationCheck(
          task.mapping, task.workload, &task.sparse_optimizations, false);
      if (task.auto_bypass_on_failure) {
        for (unsigned level = 0; level < pre_eval_status.size(); level++) {
          if (!pre_eval_status[level].success) {
            for (unsigned pvi = 0; pvi < problem::GetShape()->NumDataSpaces;
                 pvi++) {
              task.mapping.datatype_bypass_nest.at(pvi).reset(level - 1);
            }
          }
        }
      } else {
        queue_result(EvaluationResult{task.id, pre_eval_status, std::nullopt});
      }

      auto eval_status = engine.Evaluate(task.mapping, task.workload,
                                         &task.sparse_optimizations);
      for (unsigned level = 0; level < eval_status.size(); level++) {
        if (!eval_status[level].success) {
          std::cerr << "ERROR: couldn't map level " << level_names.at(level)
                    << ": " << eval_status[level].fail_reason << std::endl;
        }
      }

      if (engine.IsEvaluated()) {
        auto topology = engine.GetTopology();
        queue_result(EvaluationResult{
            task.id, pre_eval_status, eval_status, engine.Utilization(),
            engine.Energy(), engine.Area(), engine.Cycles(),
            topology.AlgorithmicComputes(), topology.ActualComputes(),
            topology.LastLevelAccesses()});
      }
    }
  }

  void queue_result(EvaluationResult eval_result) {
    {
      std::lock_guard<std::mutex> result_lock(result_mutex_);
      result_q_.push(eval_result);
    }
    result_cv_.notify_one();
  }

  void stop_workers() {
    terminate_ = true;
    worker_cv_.notify_all();

    for (auto& worker : workers_) {
      worker.join();
    }

    workers_.clear();
  }
};
}  // namespace

void BindMapperClasses(py::module& m) {
  py::class_<AcceleratorPool>(m, "NativeAcceleratorPool")
      .def(py::init<model::Engine::Specs, unsigned>())
      .def("evaluate", &AcceleratorPool::Evaluate,
           py::call_guard<py::scoped_ostream_redirect,
                          py::scoped_estream_redirect>())
      .def("get_result", &AcceleratorPool::GetResult)
      .def("try_get_reuslt", &AcceleratorPool::TryGetResult);

  py::class_<EvaluationResult>(m, "EvaluationResult")
      .def_readonly("id", &EvaluationResult::id)
      .def_readonly("pre_eval_status", &EvaluationResult::pre_eval_status)
      .def_readonly("eval_status", &EvaluationResult::eval_status)
      .def_readonly("utilization", &EvaluationResult::utilization)
      .def_readonly("energy", &EvaluationResult::energy)
      .def_readonly("area", &EvaluationResult::area)
      .def_readonly("cycles", &EvaluationResult::cycles)
      .def_readonly("algorithmic_computes",
                    &EvaluationResult::algorithmic_computes)
      .def_readonly("actual_computes", &EvaluationResult::actual_computes)
      .def_readonly("last_level_accesses",
                    &EvaluationResult::last_level_accesses);
}
