#include "bindings/model/accelerator-pool.h"

#include <pybind11/iostream.h>

#include "bindings/model/accelerator.h"

AcceleratorPool::AcceleratorPool(model::Engine::Specs arch_specs,
                                 unsigned num_workers)
    : arch_specs_(arch_specs), terminate_(false), cur_id_(0) {
  for (int i = 0; i < num_workers; i++) {
    workers_.push_back(std::thread(&AcceleratorPool::worker_loop, this));
  }
};

AcceleratorPool::~AcceleratorPool() { stop_workers(); }

uint64_t AcceleratorPool::Evaluate(
    Mapping& mapping, problem::Workload& workload,
    sparse::SparseOptimizationInfo& sparse_optimizations,
    bool break_on_failure) {
  uint64_t task_id;
  {
    std::lock_guard<std::mutex> lock(pool_mutex_);
    task_id = cur_id_++;
    task_q_.push(EvaluationTask{task_id, mapping, workload,
                                sparse_optimizations, break_on_failure});
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
  Accelerator accelerator(arch_specs_);

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

    auto result =
        accelerator.Evaluate(task.mapping, task.workload,
                             task.sparse_optimizations, task.break_on_failure);
    result.id = task.id;

    queue_result(std::move(result));
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
  py::class_<AcceleratorPool>(m, "AcceleratorPool")
      .def(py::init<model::Engine::Specs, unsigned>())
      .def("evaluate", &AcceleratorPool::Evaluate,
           py::call_guard<py::scoped_ostream_redirect,
                          py::scoped_estream_redirect>(),
           py::arg(), py::arg(), py::arg(), py::arg("break_on_failure") = false)
      .def("get_result", &AcceleratorPool::GetResult);
}
}  // namespace model_bindings
