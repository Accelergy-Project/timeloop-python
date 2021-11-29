#include "bindings/model/accelerator-pool.h"

#include <pybind11/iostream.h>

AcceleratorPool::AcceleratorPool(model::Engine::Specs arch_specs,
                                 unsigned num_workers)
    : arch_specs_(arch_specs), terminate_(false), cur_id_(0) {
  for (int i = 0; i < num_workers; i++) {
    workers_.push_back(std::thread(&AcceleratorPool::worker_loop, this));
  }
};

AcceleratorPool::~AcceleratorPool() { stop_workers(); }

uint64_t AcceleratorPool::Evaluate(
    Mapping mapping, const problem::Workload workload,
    const sparse::SparseOptimizationInfo sparse_optimizations,
    bool break_on_failure, bool auto_bypass_on_failure) {
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

EvaluationResult AcceleratorPool::GetResult() {
  std::unique_lock<std::mutex> lock(result_mutex_);
  result_cv_.wait(lock, [this]() { return !result_q_.empty() || terminate_; });

  EvaluationResult res = result_q_.front();
  result_q_.pop();
  return res;
}

void AcceleratorPool::worker_loop() {
  model::Engine engine;
  engine.Spec(arch_specs_);

  auto level_names = arch_specs_.topology.LevelNames();

  while (true) {
    // Acquire task from task_q_
    std::unique_lock<std::mutex> lock(pool_mutex_);
    worker_cv_.wait(lock, [this]() { return !task_q_.empty() || terminate_; });

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

void AcceleratorPool::queue_result(EvaluationResult eval_result) {
  {
    std::lock_guard<std::mutex> result_lock(result_mutex_);
    result_q_.push(eval_result);
  }
  result_cv_.notify_one();
}

void AcceleratorPool::stop_workers() {
  terminate_ = true;
  worker_cv_.notify_all();

  for (auto& worker : workers_) {
    worker.join();
  }

  workers_.clear();
}

namespace model_bindings {
void BindAcceleratorPool(py::module& m) {
  py::class_<AcceleratorPool>(m, "NativeAcceleratorPool")
      .def(py::init<model::Engine::Specs, unsigned>())
      .def("evaluate", &AcceleratorPool::Evaluate,
           py::call_guard<py::scoped_ostream_redirect,
                          py::scoped_estream_redirect>())
      .def("get_result", &AcceleratorPool::GetResult);
}
}  // namespace model_bindings
